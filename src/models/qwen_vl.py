"""
Qwen2-VL Model Wrapper for Text Dominance Analysis.

Supports Qwen2.5-VL and Qwen2-VL models.
"""

import torch
from typing import Dict, Optional, Any, Union, List
from PIL import Image
import requests
from io import BytesIO

from .base import BaseMLLMWrapper, ModelConfig, InferenceOutput
from ..attention import TokenMasks


class Qwen2VLWrapper(BaseMLLMWrapper):
    """
    Wrapper for Qwen2-VL and Qwen2.5-VL models.

    Supports:
    - Qwen2.5-VL-7B-Instruct
    - Qwen2.5-VL-2B-Instruct
    - Qwen2-VL-7B-Instruct
    """

    # Default vision token for Qwen2-VL
    VISION_TOKEN = "<|vision_start|>"
    DEFAULT_VISION_TOKEN_ID = 151652  # May vary by model

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    ):
        """
        Initialize Qwen2-VL wrapper.

        Args:
            config: Optional ModelConfig
            model_path: HuggingFace model path or local path
        """
        if config is None:
            config = ModelConfig(
                model_name="qwen2-vl",
                model_path=model_path,
            )
        super().__init__(config)
        self.model_path = model_path or config.model_path
        self.vision_token_id = None

    def load_model(self) -> None:
        """Load Qwen2-VL model and processor."""
        try:
            from transformers import (
                AutoProcessor,
                BitsAndBytesConfig,
            )
            # Try new class first, fallback to old
            try:
                from transformers import AutoModelForImageTextToText as AutoModelForVision
            except ImportError:
                from transformers import AutoModelForVision2Seq as AutoModelForVision
        except ImportError:
            raise ImportError(
                "Please install transformers>=4.40.0 and qwen-vl-utils"
            )

        # Setup quantization config if needed
        quantization_config = None
        if self.config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.config.get_torch_dtype(),
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif self.config.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load model with AutoModel for better compatibility
        model_kwargs = {
            "torch_dtype": self.config.get_torch_dtype(),
            "device_map": "auto" if self.config.device == "cuda" else None,
            "attn_implementation": "eager",  # Required for attention extraction
            "trust_remote_code": True,  # Required for Qwen models
        }
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config

        self.model = AutoModelForVision.from_pretrained(
            self.model_path,
            **model_kwargs
        )

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            min_pixels=256*28*28,  # Min resolution
            max_pixels=1280*28*28  # Max resolution
        )
        self.tokenizer = self.processor.tokenizer

        # Get vision token ID
        if hasattr(self.processor, 'image_token_id'):
            self.vision_token_id = self.processor.image_token_id
        elif hasattr(self.model.config, 'vision_token_id'):
            self.vision_token_id = self.model.config.vision_token_id
        else:
            self.vision_token_id = self.DEFAULT_VISION_TOKEN_ID

        self.config.image_token_id = self.vision_token_id
        self._loaded = True

    def preprocess(
        self,
        text: str,
        media: Optional[Union[Image.Image, str]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess text and image for Qwen2-VL.

        Args:
            text: Text prompt
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

        # Qwen2-VL uses conversation format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": media} if media is not None else None,
                    {"type": "text", "text": text}
                ]
            }
        ]

        # Remove None items
        messages[0]["content"] = [item for item in messages[0]["content"] if item is not None]

        # Apply chat template
        text_input = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Process inputs
        if media is not None:
            inputs = self.processor(
                text=[text_input],
                images=[media],
                padding=True,
                return_tensors="pt"
            )
        else:
            inputs = self.processor(
                text=[text_input],
                padding=True,
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
        Create token masks for Qwen2-VL.

        Vision tokens are identified by the vision_token_id.
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

        # Identify vision tokens (non-text modality)
        # Qwen2-VL uses special vision tokens
        nontext_mask = (input_ids == self.vision_token_id)

        # Also check for image_pad_token if exists
        if hasattr(self.processor, 'image_pad_token_id'):
            nontext_mask |= (input_ids == self.processor.image_pad_token_id)

        # Text tokens: everything that's not vision token
        # Exclude special tokens (padding, bos, eos)
        special_token_ids = []
        if self.tokenizer is not None:
            if self.tokenizer.pad_token_id is not None:
                special_token_ids.append(self.tokenizer.pad_token_id)
            if self.tokenizer.bos_token_id is not None:
                special_token_ids.append(self.tokenizer.bos_token_id)
            if self.tokenizer.eos_token_id is not None:
                special_token_ids.append(self.tokenizer.eos_token_id)
            # Add im_start, im_end tokens
            if hasattr(self.tokenizer, 'im_start_id'):
                special_token_ids.append(self.tokenizer.im_start_id)
            if hasattr(self.tokenizer, 'im_end_id'):
                special_token_ids.append(self.tokenizer.im_end_id)

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
        from ..attention import TokenMasks
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
        output_token_indices = torch.arange(
            input_len,
            outputs.sequences.shape[1],
            device=outputs.sequences.device
        )

        return {
            'outputs': outputs,
            'attentions': attentions,
            'token_masks': token_masks,
            'input_ids': inputs['input_ids'],
            'generated_ids': generated_ids,
            'output_token_indices': output_token_indices,
        }


class Qwen2VLAnalyzer:
    """
    High-level analyzer for Qwen2-VL models.

    Provides convenient methods for analyzing text dominance.
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
        load_in_4bit: bool = False,
    ):
        """
        Initialize analyzer.

        Args:
            model_path: Path to Qwen2-VL model
            device: Device to use
            load_in_4bit: Whether to use 4-bit quantization
        """
        config = ModelConfig(
            model_name="qwen2-vl",
            model_path=model_path,
            device=device,
            load_in_4bit=load_in_4bit,
        )
        self.wrapper = Qwen2VLWrapper(config)

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
