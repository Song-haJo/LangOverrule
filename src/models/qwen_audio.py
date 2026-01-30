"""
Qwen-Audio Model Wrapper for Text Dominance Analysis.

Supports Qwen2-Audio and similar audio-language models.
"""

import torch
from typing import Dict, Optional, Any, Union
import numpy as np

from .base import BaseMLLMWrapper, ModelConfig
from ..attention import TokenMasks


class QwenAudioWrapper(BaseMLLMWrapper):
    """
    Wrapper for Qwen-Audio models.

    Supports:
    - Qwen2-Audio-7B-Instruct
    """

    AUDIO_TOKEN = "<audio>"

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        model_path: str = "Qwen/Qwen2-Audio-7B-Instruct",
    ):
        if config is None:
            config = ModelConfig(
                model_name="qwen_audio",
                model_path=model_path,
            )
        super().__init__(config)
        self.model_path = model_path or config.model_path
        self.audio_token_id = None
        self.sample_rate = 16000

    def load_model(self) -> None:
        """Load Qwen-Audio model and processor."""
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError:
            raise ImportError(
                "Please install transformers: pip install transformers>=4.40.0"
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.config.get_torch_dtype(),
            device_map="auto" if self.config.device == "cuda" else None,
            trust_remote_code=True,
        )

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        if hasattr(self.processor, 'tokenizer'):
            self.tokenizer = self.processor.tokenizer
        else:
            self.tokenizer = self.processor

        # Get audio token ID
        if hasattr(self.model.config, 'audio_token_index'):
            self.audio_token_id = self.model.config.audio_token_index

        self._loaded = True

    def load_audio(
        self,
        audio_path: str,
        target_sr: Optional[int] = None,
    ) -> np.ndarray:
        """
        Load audio file.

        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate

        Returns:
            Audio waveform array
        """
        target_sr = target_sr or self.sample_rate

        try:
            import librosa
        except ImportError:
            raise ImportError("Please install librosa: pip install librosa")

        waveform, sr = librosa.load(audio_path, sr=target_sr)
        return waveform

    def preprocess(
        self,
        text: str,
        media: Optional[Union[str, np.ndarray]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess text and audio.

        Args:
            text: Text prompt
            media: Audio path or waveform array
            **kwargs: Additional arguments

        Returns:
            Dictionary with model inputs
        """
        audio = None

        if media is not None:
            if isinstance(media, str):
                audio = self.load_audio(media)
            elif isinstance(media, np.ndarray):
                audio = media

        # Add audio token if not present
        if audio is not None and self.AUDIO_TOKEN not in text:
            text = f"{self.AUDIO_TOKEN}\n{text}"

        # Process inputs
        if audio is not None:
            inputs = self.processor(
                text=text,
                audios=audio,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=text,
                return_tensors="pt",
            )

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
        """Create token masks for Qwen-Audio."""
        if input_ids.dim() > 1:
            input_ids = input_ids[0]

        device = input_ids.device
        seq_len = len(input_ids)

        # Audio tokens
        if self.audio_token_id is not None:
            nontext_mask = (input_ids == self.audio_token_id)
        else:
            nontext_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)

        # Special tokens
        special_token_ids = []
        if self.tokenizer is not None:
            for attr in ['pad_token_id', 'bos_token_id', 'eos_token_id']:
                token_id = getattr(self.tokenizer, attr, None)
                if token_id is not None:
                    special_token_ids.append(token_id)

        special_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        for token_id in special_token_ids:
            special_mask |= (input_ids == token_id)

        text_mask = ~nontext_mask & ~special_mask

        return TokenMasks(
            text_mask=text_mask,
            nontext_mask=nontext_mask,
        )


class QwenAudioAnalyzer:
    """High-level analyzer for Qwen-Audio models."""

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2-Audio-7B-Instruct",
        device: str = "cuda",
    ):
        config = ModelConfig(
            model_name="qwen_audio",
            model_path=model_path,
            device=device,
        )
        self.wrapper = QwenAudioWrapper(config)

    def analyze_audio(
        self,
        audio_path: str,
        question: str,
        replicate_factor: int = 1,
    ) -> Dict[str, Any]:
        """
        Analyze audio with optional token replication.

        The paper tests with ×1, ×5, ×10 replication to study
        how increasing non-text tokens affects text dominance.

        Args:
            audio_path: Path to audio file
            question: Question about the audio
            replicate_factor: Factor to replicate audio tokens (1, 5, or 10)

        Returns:
            Analysis results
        """
        if not self.wrapper._loaded:
            self.wrapper.load_model()

        from ..metrics import compute_modality_metrics

        # Load and optionally replicate audio
        audio = self.wrapper.load_audio(audio_path)
        if replicate_factor > 1:
            audio = np.tile(audio, replicate_factor)

        # Preprocess
        inputs = self.wrapper.preprocess(question, audio)

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
            'num_audio_tokens': token_masks.num_nontext_tokens,
            'num_text_tokens': token_masks.num_text_tokens,
        }
