"""
ChatTS Model Wrapper for Text Dominance Analysis in Time-Series Tasks.

Supports ChatTS and similar time-series language models.
"""

import torch
from typing import Dict, Optional, Any, Union, List
import numpy as np

from .base import BaseMLLMWrapper, ModelConfig
from ..attention import TokenMasks


class ChatTSWrapper(BaseMLLMWrapper):
    """
    Wrapper for ChatTS (time-series + LLM) models.

    ChatTS aligns time series with LLMs via synthetic data
    for enhanced understanding and reasoning.
    """

    TIMESERIES_TOKEN = "<ts>"

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        model_path: str = "ChatTS-14B",  # Placeholder - adjust to actual path
    ):
        if config is None:
            config = ModelConfig(
                model_name="chatts",
                model_path=model_path,
            )
        super().__init__(config)
        self.model_path = model_path or config.model_path
        self.ts_token_id = None

    def load_model(self) -> None:
        """Load ChatTS model."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "Please install transformers: pip install transformers>=4.40.0"
            )

        # Note: ChatTS may require custom model loading
        # This is a placeholder implementation
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.config.get_torch_dtype(),
            device_map="auto" if self.config.device == "cuda" else None,
            trust_remote_code=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        # Get time-series token ID (model-specific)
        if hasattr(self.model.config, 'ts_token_index'):
            self.ts_token_id = self.model.config.ts_token_index

        self._loaded = True

    def encode_timeseries(
        self,
        data: np.ndarray,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Encode time-series data for the model.

        Args:
            data: Time-series data array (T,) or (T, D) for multivariate
            normalize: Whether to normalize the data

        Returns:
            Encoded tensor
        """
        if normalize:
            mean = np.mean(data)
            std = np.std(data) + 1e-8
            data = (data - mean) / std

        return torch.from_numpy(data).float()

    def preprocess(
        self,
        text: str,
        media: Optional[Union[np.ndarray, torch.Tensor]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess text and time-series data.

        Args:
            text: Text prompt with task description
            media: Time-series data array
            **kwargs: Additional arguments

        Returns:
            Dictionary with model inputs
        """
        ts_data = None

        if media is not None:
            if isinstance(media, np.ndarray):
                ts_data = self.encode_timeseries(media)
            elif isinstance(media, torch.Tensor):
                ts_data = media

        # Add time-series token if not present
        if ts_data is not None and self.TIMESERIES_TOKEN not in text:
            text = f"{self.TIMESERIES_TOKEN}\n{text}"

        # Tokenize text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
        )

        # Add time-series embeddings (model-specific)
        if ts_data is not None:
            inputs['timeseries_values'] = ts_data.unsqueeze(0)

        # Move to device
        device = self.config.device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        return inputs

    def get_token_masks(
        self,
        input_ids: torch.Tensor,
        **kwargs
    ) -> TokenMasks:
        """Create token masks for ChatTS."""
        if input_ids.dim() > 1:
            input_ids = input_ids[0]

        device = input_ids.device
        seq_len = len(input_ids)

        # Time-series tokens
        if self.ts_token_id is not None:
            nontext_mask = (input_ids == self.ts_token_id)
        else:
            # Estimate based on time-series values length
            if 'timeseries_values' in kwargs:
                ts_len = kwargs['timeseries_values'].shape[-1]
                # Assume TS tokens are at the beginning after special tokens
                nontext_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
                # This is an approximation
            else:
                nontext_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)

        # Special tokens
        special_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        if self.tokenizer is not None:
            for attr in ['pad_token_id', 'bos_token_id', 'eos_token_id']:
                token_id = getattr(self.tokenizer, attr, None)
                if token_id is not None:
                    special_mask |= (input_ids == token_id)

        text_mask = ~nontext_mask & ~special_mask

        return TokenMasks(
            text_mask=text_mask,
            nontext_mask=nontext_mask,
        )


class ChatTSAnalyzer:
    """High-level analyzer for ChatTS models."""

    def __init__(
        self,
        model_path: str = "ChatTS-14B",
        device: str = "cuda",
    ):
        config = ModelConfig(
            model_name="chatts",
            model_path=model_path,
            device=device,
        )
        self.wrapper = ChatTSWrapper(config)

    def analyze_timeseries(
        self,
        timeseries: np.ndarray,
        question: str,
        replicate_factor: int = 1,
    ) -> Dict[str, Any]:
        """
        Analyze time-series data with optional token replication.

        Args:
            timeseries: Time-series data array
            question: Question/task about the time-series
            replicate_factor: Factor to replicate TS tokens (1, 5, or 10)

        Returns:
            Analysis results
        """
        if not self.wrapper._loaded:
            self.wrapper.load_model()

        from ..metrics import compute_modality_metrics

        # Replicate time-series if needed
        if replicate_factor > 1:
            timeseries = np.tile(timeseries, replicate_factor)

        # Preprocess
        inputs = self.wrapper.preprocess(question, timeseries)

        # Enable attention output
        self.wrapper.model.config.output_attentions = True

        # Forward pass
        with torch.no_grad():
            outputs = self.wrapper.model(**inputs, output_attentions=True)

        # Get masks and attention
        token_masks = self.wrapper.get_token_masks(inputs['input_ids'], **inputs)

        attentions = []
        if hasattr(outputs, 'attentions') and outputs.attentions:
            attentions = list(outputs.attentions)

        # Compute metrics
        metrics = {}
        if attentions:
            metrics = compute_modality_metrics(
                attentions,
                token_masks.text_mask,
                token_masks.nontext_mask,
                layerwise=True,
            )

        return {
            'metrics': metrics,
            'replicate_factor': replicate_factor,
            'num_ts_tokens': token_masks.num_nontext_tokens,
            'num_text_tokens': token_masks.num_text_tokens,
        }

    def generate_synthetic_timeseries(
        self,
        length: int = 100,
        pattern: str = "sine",
        noise_level: float = 0.1,
    ) -> np.ndarray:
        """
        Generate synthetic time-series for testing.

        Args:
            length: Length of time-series
            pattern: Pattern type ('sine', 'trend', 'step', 'random')
            noise_level: Amount of noise to add

        Returns:
            Generated time-series array
        """
        t = np.linspace(0, 4 * np.pi, length)
        noise = np.random.randn(length) * noise_level

        if pattern == "sine":
            data = np.sin(t) + noise
        elif pattern == "trend":
            data = t / (4 * np.pi) + noise
        elif pattern == "step":
            data = np.where(t > 2 * np.pi, 1.0, 0.0) + noise
        else:  # random
            data = np.random.randn(length)

        return data
