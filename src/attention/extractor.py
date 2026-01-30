"""
Attention Extraction Utilities for Transformer-based MLLMs.

This module provides tools to extract attention weights from various
transformer architectures during forward pass using PyTorch hooks.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from contextlib import contextmanager
import re


@dataclass
class AttentionHook:
    """Container for storing attention weights captured by hooks."""
    layer_name: str
    attention_weights: Optional[torch.Tensor] = None
    handle: Optional[Any] = None

    def clear(self):
        """Clear stored attention weights."""
        self.attention_weights = None


class AttentionExtractor:
    """
    Extracts attention weights from transformer models.

    Supports various MLLM architectures by automatically detecting
    attention layers and registering hooks to capture attention weights.
    """

    # Common attention layer patterns for different architectures
    ATTENTION_PATTERNS = {
        'llama': r'.*\.self_attn$',
        'llava': r'.*\.self_attn$',
        'qwen': r'.*\.attn$',
        'qwen2': r'.*\.self_attn$',
        'mistral': r'.*\.self_attn$',
        'vicuna': r'.*\.self_attn$',
        'video_llama': r'.*\.self_attn$',
        'default': r'.*\.(self_attn|attention|attn)$',
    }

    def __init__(
        self,
        model: nn.Module,
        model_type: str = 'default',
        layer_pattern: Optional[str] = None,
    ):
        """
        Initialize the attention extractor.

        Args:
            model: The transformer model to extract attention from
            model_type: Type of model architecture ('llama', 'llava', 'qwen', etc.)
            layer_pattern: Custom regex pattern for matching attention layers
        """
        self.model = model
        self.model_type = model_type.lower()
        self.hooks: Dict[str, AttentionHook] = {}
        self.layer_pattern = layer_pattern or self.ATTENTION_PATTERNS.get(
            self.model_type, self.ATTENTION_PATTERNS['default']
        )
        self._registered = False

    def _find_attention_layers(self) -> List[Tuple[str, nn.Module]]:
        """Find all attention layers matching the pattern."""
        attention_layers = []
        pattern = re.compile(self.layer_pattern)

        for name, module in self.model.named_modules():
            if pattern.match(name):
                attention_layers.append((name, module))

        return attention_layers

    def _create_hook(self, layer_name: str) -> Callable:
        """Create a forward hook function to capture attention weights."""

        def hook_fn(module, input, output):
            # Handle different output formats
            if isinstance(output, tuple):
                # Most transformers return (hidden_states, attention_weights, ...)
                # or (hidden_states, present_key_value, attention_weights)
                for item in output:
                    if isinstance(item, torch.Tensor):
                        # Check if this looks like attention weights
                        # Attention: (batch, heads, seq, seq) or (batch, seq, seq)
                        if item.dim() >= 3:
                            shape = item.shape
                            if len(shape) >= 3 and shape[-1] == shape[-2]:
                                self.hooks[layer_name].attention_weights = item.detach()
                                return
            elif isinstance(output, torch.Tensor):
                if output.dim() >= 3:
                    shape = output.shape
                    if shape[-1] == shape[-2]:
                        self.hooks[layer_name].attention_weights = output.detach()

        return hook_fn

    def _create_attention_hook_for_hf(self, layer_name: str) -> Callable:
        """
        Create hook specifically for HuggingFace transformers that need
        output_attentions=True to return attention weights.
        """

        def hook_fn(module, input, output):
            # For HF models with output_attentions=True
            # Output is typically (hidden_states, attention_weights) or
            # (hidden_states, present_key_value, attention_weights)
            if isinstance(output, tuple) and len(output) >= 2:
                # Try to find attention weights in the output
                for i, item in enumerate(output):
                    if isinstance(item, torch.Tensor):
                        if item.dim() == 4:  # (batch, heads, seq, seq)
                            self.hooks[layer_name].attention_weights = item.detach()
                            return

        return hook_fn

    def register_hooks(self, use_hf_style: bool = True) -> int:
        """
        Register forward hooks on all attention layers.

        Args:
            use_hf_style: Use HuggingFace-specific hook style

        Returns:
            Number of hooks registered
        """
        if self._registered:
            self.remove_hooks()

        attention_layers = self._find_attention_layers()

        for name, module in attention_layers:
            hook = AttentionHook(layer_name=name)

            if use_hf_style:
                hook_fn = self._create_attention_hook_for_hf(name)
            else:
                hook_fn = self._create_hook(name)

            handle = module.register_forward_hook(hook_fn)
            hook.handle = handle
            self.hooks[name] = hook

        self._registered = True
        return len(self.hooks)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks.values():
            if hook.handle is not None:
                hook.handle.remove()
        self.hooks.clear()
        self._registered = False

    def clear_attention(self):
        """Clear all stored attention weights."""
        for hook in self.hooks.values():
            hook.clear()

    def get_attention_weights(self) -> Dict[str, torch.Tensor]:
        """
        Get all captured attention weights.

        Returns:
            Dictionary mapping layer names to attention tensors
        """
        return {
            name: hook.attention_weights
            for name, hook in self.hooks.items()
            if hook.attention_weights is not None
        }

    def get_attention_list(self) -> List[torch.Tensor]:
        """
        Get attention weights as a list ordered by layer.

        Returns:
            List of attention tensors in layer order
        """
        weights = self.get_attention_weights()
        # Sort by layer number if possible
        sorted_names = sorted(
            weights.keys(),
            key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0
        )
        return [weights[name] for name in sorted_names]

    @contextmanager
    def capture(self):
        """
        Context manager for capturing attention weights.

        Usage:
            extractor = AttentionExtractor(model)
            with extractor.capture():
                output = model(input)
            attention_weights = extractor.get_attention_weights()
        """
        self.register_hooks()
        try:
            yield self
        finally:
            pass  # Keep hooks for getting weights

    def __enter__(self):
        self.register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()
        return False


class HFAttentionExtractor(AttentionExtractor):
    """
    Specialized extractor for HuggingFace transformers.

    Automatically configures model to output attention weights.
    """

    def __init__(self, model: nn.Module, model_type: str = 'default'):
        super().__init__(model, model_type)
        self._original_config = {}

    def enable_attention_output(self):
        """Enable attention output in model config."""
        if hasattr(self.model, 'config'):
            self._original_config['output_attentions'] = getattr(
                self.model.config, 'output_attentions', False
            )
            self.model.config.output_attentions = True

    def restore_config(self):
        """Restore original model config."""
        if hasattr(self.model, 'config') and 'output_attentions' in self._original_config:
            self.model.config.output_attentions = self._original_config['output_attentions']

    @contextmanager
    def capture(self):
        """Capture with automatic attention output enabling."""
        self.enable_attention_output()
        self.register_hooks()
        try:
            yield self
        finally:
            self.restore_config()


def extract_attention_from_output(
    model_output: Any,
    return_list: bool = True
) -> List[torch.Tensor]:
    """
    Extract attention weights from HuggingFace model output.

    Many HF models return attention weights directly in their output
    when output_attentions=True.

    Args:
        model_output: Output from model forward pass
        return_list: If True, return as list; otherwise return dict

    Returns:
        List or dict of attention weight tensors
    """
    attentions = []

    # Check for attentions attribute (common in HF outputs)
    if hasattr(model_output, 'attentions') and model_output.attentions is not None:
        attentions = list(model_output.attentions)

    # Check for cross_attentions
    if hasattr(model_output, 'cross_attentions') and model_output.cross_attentions is not None:
        attentions.extend(list(model_output.cross_attentions))

    # Check tuple output
    if isinstance(model_output, tuple):
        for item in model_output:
            if isinstance(item, tuple) and len(item) > 0:
                if isinstance(item[0], torch.Tensor) and item[0].dim() == 4:
                    attentions = list(item)
                    break

    return attentions
