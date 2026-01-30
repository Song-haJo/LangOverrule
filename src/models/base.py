"""
Base wrapper class for MLLM models.

Provides unified interface for loading models, extracting attention,
and computing modality dominance metrics.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path

from ..attention import AttentionExtractor, AttentionAnalyzer, TokenMasks
from ..metrics import ModalityMetrics, compute_modality_metrics


@dataclass
class ModelConfig:
    """Configuration for MLLM wrapper."""
    model_name: str
    model_path: Optional[str] = None
    device: str = "cuda"
    torch_dtype: str = "float16"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_flash_attention: bool = False
    max_memory: Optional[Dict[int, str]] = None

    # Token IDs (model-specific)
    image_token_id: Optional[int] = None
    video_token_id: Optional[int] = None
    audio_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None

    def get_torch_dtype(self) -> torch.dtype:
        """Convert string dtype to torch dtype."""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.torch_dtype, torch.float16)


@dataclass
class InferenceOutput:
    """Container for inference output with attention."""
    generated_text: str
    generated_ids: torch.Tensor
    attention_weights: List[torch.Tensor]
    token_masks: TokenMasks
    input_ids: torch.Tensor
    metrics: Optional[Dict[str, ModalityMetrics]] = None


class BaseMLLMWrapper(ABC):
    """
    Abstract base class for MLLM wrappers.

    Subclasses must implement model-specific loading and preprocessing.
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize wrapper.

        Args:
            config: ModelConfig with model settings
        """
        self.config = config
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.attention_extractor = None
        self.analyzer = AttentionAnalyzer()
        self._loaded = False

    @abstractmethod
    def load_model(self) -> None:
        """Load the model and processor. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def preprocess(
        self,
        text: str,
        media: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess inputs for the model.

        Args:
            text: Text prompt
            media: Optional media input (image, video, audio, etc.)
            **kwargs: Additional preprocessing arguments

        Returns:
            Dictionary of model inputs
        """
        pass

    @abstractmethod
    def get_token_masks(
        self,
        input_ids: torch.Tensor,
        **kwargs
    ) -> TokenMasks:
        """
        Create token masks for modality separation.

        Args:
            input_ids: Token IDs tensor
            **kwargs: Additional arguments

        Returns:
            TokenMasks object
        """
        pass

    def setup_attention_extraction(self) -> None:
        """Setup attention extraction hooks."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        self.attention_extractor = AttentionExtractor(
            self.model,
            model_type=self.config.model_name.lower()
        )

    @torch.no_grad()
    def generate_with_attention(
        self,
        text: str,
        media: Optional[Any] = None,
        max_new_tokens: int = 128,
        output_attentions: bool = True,
        **generate_kwargs
    ) -> InferenceOutput:
        """
        Generate text while capturing attention weights.

        Args:
            text: Input text prompt
            media: Optional media input
            max_new_tokens: Maximum tokens to generate
            output_attentions: Whether to output attention weights
            **generate_kwargs: Additional generation arguments

        Returns:
            InferenceOutput with generated text and attention data
        """
        if not self._loaded:
            self.load_model()
            self._loaded = True

        # Preprocess inputs
        inputs = self.preprocess(text, media)
        input_ids = inputs['input_ids']

        # Get token masks
        token_masks = self.get_token_masks(input_ids, **inputs)

        # Enable attention output
        if hasattr(self.model, 'config'):
            self.model.config.output_attentions = output_attentions

        # Generate with attention
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            output_attentions=output_attentions,
            return_dict_in_generate=True,
            **generate_kwargs
        )

        # Extract attention weights
        attention_weights = self._extract_attention_from_output(outputs)

        # Decode generated text
        generated_ids = outputs.sequences
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            generated_text = self.tokenizer.decode(
                generated_ids[0][input_ids.shape[1]:],
                skip_special_tokens=True
            )
        else:
            generated_text = ""

        # Compute metrics if attention available
        metrics = None
        if attention_weights:
            metrics = compute_modality_metrics(
                attention_weights,
                token_masks.text_mask,
                token_masks.nontext_mask,
                layerwise=True,
            )

        return InferenceOutput(
            generated_text=generated_text,
            generated_ids=generated_ids,
            attention_weights=attention_weights,
            token_masks=token_masks,
            input_ids=input_ids,
            metrics=metrics,
        )

    def _extract_attention_from_output(
        self,
        outputs: Any
    ) -> List[torch.Tensor]:
        """Extract attention weights from model output."""
        attention_weights = []

        # Try different output formats
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            # Standard HF format
            for attn in outputs.attentions:
                if isinstance(attn, tuple):
                    attention_weights.extend([a for a in attn if a is not None])
                elif attn is not None:
                    attention_weights.append(attn)

        elif hasattr(outputs, 'decoder_attentions') and outputs.decoder_attentions:
            attention_weights = list(outputs.decoder_attentions)

        return attention_weights

    def analyze_attention(
        self,
        attention_weights: List[torch.Tensor],
        token_masks: TokenMasks,
    ) -> Dict[str, ModalityMetrics]:
        """
        Analyze attention distribution.

        Args:
            attention_weights: List of attention tensors
            token_masks: Token masks for modality separation

        Returns:
            Dictionary of metrics by layer stage
        """
        return compute_modality_metrics(
            attention_weights,
            token_masks.text_mask,
            token_masks.nontext_mask,
            layerwise=True,
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = {
            "model_name": self.config.model_name,
            "loaded": self._loaded,
        }

        if self._loaded and self.model is not None:
            info["num_parameters"] = sum(p.numel() for p in self.model.parameters())
            if hasattr(self.model, 'config'):
                info["num_layers"] = getattr(self.model.config, 'num_hidden_layers', None)
                info["hidden_size"] = getattr(self.model.config, 'hidden_size', None)

        return info

    def to(self, device: Union[str, torch.device]) -> 'BaseMLLMWrapper':
        """Move model to device."""
        if self.model is not None:
            self.model.to(device)
        self.config.device = str(device)
        return self

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.config.model_name}, loaded={self._loaded})"
