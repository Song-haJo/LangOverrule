"""
LLaVA Model Wrapper for Text Dominance Analysis.

Supports LLaVA-1.5 and similar architectures.
"""

import torch
from typing import Dict, Optional, Any, Union, List
from PIL import Image
import requests
from io import BytesIO

from .base import BaseMLLMWrapper, ModelConfig, InferenceOutput
from ..attention import TokenMasks


class LLaVAWrapper(BaseMLLMWrapper):
    """
    Wrapper for LLaVA models.

    Supports:
    - LLaVA-1.5-7B
    - LLaVA-1.5-13B
    - LLaVA-1.6-vicuna-7B
    """

    # Default image token for LLaVA
    IMAGE_TOKEN = "<image>"
    DEFAULT_IMAGE_TOKEN_ID = 32000  # May vary by model

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        model_path: str = "llava-hf/llava-1.5-7b-hf",
    ):
        """
        Initialize LLaVA wrapper.

        Args:
            config: Optional ModelConfig
            model_path: HuggingFace model path or local path
        """
        if config is None:
            config = ModelConfig(
                model_name="llava",
                model_path=model_path,
            )
        super().__init__(config)
        self.model_path = model_path or config.model_path
        self.image_token_id = None

    def load_model(self) -> None:
        """Load LLaVA model and processor."""
        try:
            from transformers import (
                LlavaForConditionalGeneration,
                AutoProcessor,
                BitsAndBytesConfig,
            )
        except ImportError:
            raise ImportError(
                "Please install transformers: pip install transformers>=4.40.0"
            )

        # Setup quantization config if needed
        quantization_config = None
        if self.config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.config.get_torch_dtype(),
            )
        elif self.config.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load model
        model_kwargs = {
            "torch_dtype": self.config.get_torch_dtype(),
            "device_map": "auto" if self.config.device == "cuda" else None,
            "attn_implementation": "eager",  # Required for attention extraction
        }
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config

        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_path,
            **model_kwargs
        )

        # Load processor
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.tokenizer = self.processor.tokenizer

        # Get image token ID
        if hasattr(self.model.config, 'image_token_index'):
            self.image_token_id = self.model.config.image_token_index
        else:
            self.image_token_id = self.DEFAULT_IMAGE_TOKEN_ID

        self.config.image_token_id = self.image_token_id
        self._loaded = True

    def preprocess(
        self,
        text: str,
        media: Optional[Union[Image.Image, str]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess text and image for LLaVA.

        Args:
            text: Text prompt (should contain <image> placeholder if using image)
            media: PIL Image or URL string
            **kwargs: Additional arguments

        Returns:
            Dictionary with input_ids, attention_mask, pixel_values
        """
        # Load image if URL provided
        if isinstance(media, str):
            if media.startswith(('http://', 'https://')):
                response = requests.get(media)
                media = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                media = Image.open(media).convert('RGB')

        # Add image token if not present and image provided
        if media is not None and self.IMAGE_TOKEN not in text:
            text = f"{self.IMAGE_TOKEN}\n{text}"

        # Process inputs
        if media is not None:
            inputs = self.processor(
                text=text,
                images=media,
                return_tensors="pt"
            )
        else:
            inputs = self.processor(
                text=text,
                return_tensors="pt"
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
        Create token masks for LLaVA.

        Image tokens are identified by the image_token_id.
        Text tokens are all other non-special tokens.

        Args:
            input_ids: Token IDs tensor
            **kwargs: May include pixel_values to verify image presence

        Returns:
            TokenMasks object
        """
        if input_ids.dim() > 1:
            input_ids = input_ids[0]

        device = input_ids.device
        seq_len = len(input_ids)

        # Identify image tokens (non-text modality)
        nontext_mask = (input_ids == self.image_token_id)

        # For LLaVA, image tokens get expanded during forward pass
        # We need to account for the expanded image tokens
        if 'pixel_values' in kwargs and kwargs['pixel_values'] is not None:
            # Check if model has vision config for patch calculation
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'vision_config'):
                vision_config = self.model.config.vision_config
                if hasattr(vision_config, 'image_size') and hasattr(vision_config, 'patch_size'):
                    image_size = vision_config.image_size
                    patch_size = vision_config.patch_size
                    num_patches = (image_size // patch_size) ** 2
                    # Update expected number of image tokens
                    # This is an approximation; actual may vary

        # Text tokens: everything that's not image token
        # Exclude special tokens (padding, bos, eos)
        special_token_ids = []
        if self.tokenizer is not None:
            if self.tokenizer.pad_token_id is not None:
                special_token_ids.append(self.tokenizer.pad_token_id)
            if self.tokenizer.bos_token_id is not None:
                special_token_ids.append(self.tokenizer.bos_token_id)
            if self.tokenizer.eos_token_id is not None:
                special_token_ids.append(self.tokenizer.eos_token_id)

        special_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        for token_id in special_token_ids:
            special_mask |= (input_ids == token_id)

        text_mask = ~nontext_mask & ~special_mask

        return TokenMasks(
            text_mask=text_mask,
            nontext_mask=nontext_mask,
        )

    @torch.no_grad()
    def forward_with_attention(
        self,
        text: str,
        image: Optional[Union[Image.Image, str]] = None,
    ) -> Dict[str, Any]:
        """
        Run forward pass and capture attention weights.

        This method performs a single forward pass (no generation)
        and captures attention from all layers.

        Args:
            text: Input text
            image: Optional image

        Returns:
            Dictionary with outputs and attention
        """
        if not self._loaded:
            self.load_model()
            self._loaded = True

        inputs = self.preprocess(text, image)

        # Enable attention output
        self.model.config.output_attentions = True

        # Forward pass
        outputs = self.model(**inputs, output_attentions=True)

        # Get token masks
        token_masks = self.get_token_masks(**inputs)

        # Extract attention and process
        attentions = []
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            for attn in outputs.attentions:
                # attn shape: [batch, heads, seq, seq]
                # Average across heads and remove batch dimension
                if attn.dim() == 4:
                    attn = attn.mean(dim=1).squeeze(0)  # -> [seq, seq]
                elif attn.dim() == 3:
                    attn = attn.mean(dim=0)  # -> [seq, seq]
                # Move to CPU immediately to avoid GPU memory accumulation
                attentions.append(attn.cpu())

        # Move token masks to CPU to match attention weights
        token_masks_cpu = TokenMasks(
            text_mask=token_masks.text_mask.cpu(),
            nontext_mask=token_masks.nontext_mask.cpu(),
        )

        return {
            'outputs': outputs,
            'attentions': attentions,
            'token_masks': token_masks_cpu,
            'input_ids': inputs['input_ids'],
        }

    def generate_with_attention(
        self,
        text: str,
        image: Optional[Union[Image.Image, str]] = None,
        max_new_tokens: int = 20,
    ) -> Dict[str, Any]:
        """
        Generate text and capture attention during generation (like the paper).

        Args:
            text: Input text
            image: Optional image
            max_new_tokens: Maximum number of tokens to generate

        Returns:
            Dictionary with outputs, attentions, and token masks
        """
        if not self._loaded:
            self.load_model()
            self._loaded = True

        inputs = self.preprocess(text, image)

        # Get input length for masking
        input_len = inputs['input_ids'].shape[1]

        # Enable attention output
        self.model.config.output_attentions = True

        # Generate with attention
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            output_attentions=True,
            return_dict_in_generate=True,
            do_sample=False,  # Greedy decoding like paper
        )

        # Get token masks from input
        token_masks = self.get_token_masks(**inputs)

        # Extract and process attentions
        # outputs.attentions is a tuple of tuples: (generation_step, layer)
        # Use only first generation step (input encoding) for consistent tensor sizes
        attentions = []
        if hasattr(outputs, 'attentions') and outputs.attentions is not None and len(outputs.attentions) > 0:
            first_step_attns = outputs.attentions[0]
            for attn in first_step_attns:
                # Average across heads and remove batch
                if attn.dim() == 4:
                    attn = attn.mean(dim=1).squeeze(0)  # -> [seq, seq]
                elif attn.dim() == 3:
                    attn = attn.mean(dim=0)  # -> [seq, seq]
                # Move to CPU immediately to avoid GPU memory accumulation
                attentions.append(attn.cpu())

        # Generated output token indices (excluding input)
        generated_ids = outputs.sequences[0, input_len:]
        # Create on CPU to match attention weights device
        output_token_indices = torch.arange(
            input_len,
            outputs.sequences.shape[1],
            device='cpu'
        )

        # Move token masks to CPU to match attention weights
        token_masks_cpu = TokenMasks(
            text_mask=token_masks.text_mask.cpu(),
            nontext_mask=token_masks.nontext_mask.cpu(),
        )

        return {
            'outputs': outputs,
            'attentions': attentions,
            'token_masks': token_masks_cpu,
            'input_ids': inputs['input_ids'],
            'generated_ids': generated_ids,
            'output_token_indices': output_token_indices,
        }


class LLaVAAnalyzer:
    """
    High-level analyzer for LLaVA models.

    Provides convenient methods for analyzing text dominance.
    """

    def __init__(
        self,
        model_path: str = "llava-hf/llava-1.5-7b-hf",
        device: str = "cuda",
        load_in_4bit: bool = False,
    ):
        """
        Initialize analyzer.

        Args:
            model_path: Path to LLaVA model
            device: Device to use
            load_in_4bit: Whether to use 4-bit quantization
        """
        config = ModelConfig(
            model_name="llava",
            model_path=model_path,
            device=device,
            load_in_4bit=load_in_4bit,
        )
        self.wrapper = LLaVAWrapper(config)

    def analyze_sample(
        self,
        text: str,
        image: Union[Image.Image, str],
    ) -> Dict[str, Any]:
        """
        Analyze a single image-text sample.

        Args:
            text: Text prompt
            image: Image (PIL Image or path/URL)

        Returns:
            Analysis results including MDI and AEI
        """
        from ..metrics import compute_modality_metrics

        result = self.wrapper.forward_with_attention(text, image)

        if result['attentions']:
            metrics = compute_modality_metrics(
                result['attentions'],
                result['token_masks'].text_mask,
                result['token_masks'].nontext_mask,
                layerwise=True,
            )
        else:
            metrics = {}

        return {
            'metrics': metrics,
            'token_masks': result['token_masks'],
            'num_layers': len(result['attentions']),
        }

    def analyze_dataset(
        self,
        samples: List[Dict[str, Any]],
        progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze multiple samples and aggregate results.

        Args:
            samples: List of dicts with 'text' and 'image' keys
            progress: Whether to show progress bar

        Returns:
            Aggregated analysis results
        """
        from tqdm import tqdm
        import numpy as np

        all_metrics = {'early': [], 'middle': [], 'late': []}

        iterator = tqdm(samples) if progress else samples
        for sample in iterator:
            result = self.analyze_sample(sample['text'], sample['image'])

            for stage in ['early', 'middle', 'late']:
                if stage in result['metrics']:
                    all_metrics[stage].append(result['metrics'][stage])

        # Aggregate
        aggregated = {}
        for stage, metrics_list in all_metrics.items():
            if metrics_list:
                aggregated[stage] = {
                    'mdi_mean': np.mean([m.mdi for m in metrics_list]),
                    'mdi_std': np.std([m.mdi for m in metrics_list]),
                    'aei_mean': np.mean([m.aei_text for m in metrics_list]),
                    'aei_std': np.std([m.aei_text for m in metrics_list]),
                    'num_samples': len(metrics_list),
                }

        return aggregated
