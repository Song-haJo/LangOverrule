"""
VideoLLaMA Model Wrapper for Text Dominance Analysis.

Supports VideoLLaMA and similar video-language models.
"""

import torch
from typing import Dict, Optional, Any, Union, List
from pathlib import Path
import numpy as np

from .base import BaseMLLMWrapper, ModelConfig
from ..attention import TokenMasks


class VideoLLaMAWrapper(BaseMLLMWrapper):
    """
    Wrapper for VideoLLaMA models.

    Supports:
    - VideoLLaMA2
    - VideoLLaMA3
    """

    VIDEO_TOKEN = "<video>"
    FRAME_TOKEN = "<frame>"

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        model_path: str = "DAMO-NLP-SG/VideoLLaMA2-7B",
    ):
        """
        Initialize VideoLLaMA wrapper.

        Args:
            config: Optional ModelConfig
            model_path: HuggingFace model path or local path
        """
        if config is None:
            config = ModelConfig(
                model_name="video_llama",
                model_path=model_path,
            )
        super().__init__(config)
        self.model_path = model_path or config.model_path
        self.video_token_id = None
        self.num_frames = 8  # Default number of frames to sample

    def load_model(self) -> None:
        """Load VideoLLaMA model and processor."""
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError:
            raise ImportError(
                "Please install transformers: pip install transformers>=4.40.0"
            )

        # Note: VideoLLaMA may require custom loading depending on version
        # This is a generic implementation; adjust for specific VideoLLaMA version

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

        # Get video token ID (model-specific)
        if hasattr(self.model.config, 'video_token_index'):
            self.video_token_id = self.model.config.video_token_index
        elif hasattr(self.model.config, 'image_token_index'):
            # Some models use image token for video frames
            self.video_token_id = self.model.config.image_token_index

        self._loaded = True

    def load_video(
        self,
        video_path: str,
        num_frames: Optional[int] = None,
    ) -> np.ndarray:
        """
        Load and sample frames from video.

        Args:
            video_path: Path to video file
            num_frames: Number of frames to sample

        Returns:
            Array of sampled frames
        """
        num_frames = num_frames or self.num_frames

        try:
            import decord
            from decord import VideoReader, cpu
        except ImportError:
            raise ImportError("Please install decord: pip install decord")

        decord.bridge.set_bridge('native')
        vr = VideoReader(video_path, ctx=cpu(0))

        total_frames = len(vr)
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()

        return frames

    def preprocess(
        self,
        text: str,
        media: Optional[Union[str, np.ndarray, List]] = None,
        num_frames: Optional[int] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess text and video for VideoLLaMA.

        Args:
            text: Text prompt
            media: Video path, frames array, or list of frame images
            num_frames: Number of frames to use
            **kwargs: Additional arguments

        Returns:
            Dictionary with model inputs
        """
        frames = None

        if media is not None:
            if isinstance(media, str):
                # Load video from path
                frames = self.load_video(media, num_frames)
            elif isinstance(media, np.ndarray):
                frames = media
            elif isinstance(media, list):
                # List of PIL images or arrays
                frames = np.stack([np.array(f) for f in media])

        # Add video token if not present
        if frames is not None and self.VIDEO_TOKEN not in text:
            text = f"{self.VIDEO_TOKEN}\n{text}"

        # Process inputs (model-specific)
        if frames is not None:
            inputs = self.processor(
                text=text,
                videos=frames,
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
        """
        Create token masks for VideoLLaMA.

        Video frame tokens are identified by video_token_id.

        Args:
            input_ids: Token IDs tensor
            **kwargs: Additional arguments

        Returns:
            TokenMasks object
        """
        if input_ids.dim() > 1:
            input_ids = input_ids[0]

        device = input_ids.device
        seq_len = len(input_ids)

        # Video tokens (non-text modality)
        if self.video_token_id is not None:
            nontext_mask = (input_ids == self.video_token_id)
        else:
            # Fallback: assume first portion after special tokens is video
            nontext_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)

        # Get special token IDs
        special_token_ids = []
        if self.tokenizer is not None:
            if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id:
                special_token_ids.append(self.tokenizer.pad_token_id)
            if hasattr(self.tokenizer, 'bos_token_id') and self.tokenizer.bos_token_id:
                special_token_ids.append(self.tokenizer.bos_token_id)
            if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id:
                special_token_ids.append(self.tokenizer.eos_token_id)

        special_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        for token_id in special_token_ids:
            special_mask |= (input_ids == token_id)

        text_mask = ~nontext_mask & ~special_mask

        return TokenMasks(
            text_mask=text_mask,
            nontext_mask=nontext_mask,
        )


class VideoLLaMAAnalyzer:
    """High-level analyzer for VideoLLaMA models."""

    def __init__(
        self,
        model_path: str = "DAMO-NLP-SG/VideoLLaMA2-7B",
        device: str = "cuda",
        num_frames: int = 8,
    ):
        config = ModelConfig(
            model_name="video_llama",
            model_path=model_path,
            device=device,
        )
        self.wrapper = VideoLLaMAWrapper(config)
        self.wrapper.num_frames = num_frames

    def analyze_video(
        self,
        video_path: str,
        question: str,
    ) -> Dict[str, Any]:
        """
        Analyze a video with a question.

        Args:
            video_path: Path to video file
            question: Question about the video

        Returns:
            Analysis results with MDI and AEI metrics
        """
        if not self.wrapper._loaded:
            self.wrapper.load_model()

        from ..metrics import compute_modality_metrics

        # Preprocess
        inputs = self.wrapper.preprocess(question, video_path)

        # Enable attention output
        self.wrapper.model.config.output_attentions = True

        # Forward pass
        with torch.no_grad():
            outputs = self.wrapper.model(**inputs, output_attentions=True)

        # Get attention and masks
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
            'token_masks': token_masks,
            'num_video_tokens': token_masks.num_nontext_tokens,
            'num_text_tokens': token_masks.num_text_tokens,
        }
