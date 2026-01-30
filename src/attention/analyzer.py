"""
Attention Analysis Utilities for Cross-Modal Attention Distribution.

This module provides tools to analyze attention patterns between
text and non-text modalities in MLLMs.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from ..metrics import MDI, AEI, ModalityMetrics, compute_modality_metrics


@dataclass
class TokenMasks:
    """Container for modality-specific token masks."""
    text_mask: torch.Tensor
    nontext_mask: torch.Tensor
    output_mask: Optional[torch.Tensor] = None
    system_mask: Optional[torch.Tensor] = None

    @property
    def total_tokens(self) -> int:
        return len(self.text_mask)

    @property
    def num_text_tokens(self) -> int:
        return self.text_mask.sum().item()

    @property
    def num_nontext_tokens(self) -> int:
        return self.nontext_mask.sum().item()

    def to(self, device: torch.device) -> 'TokenMasks':
        """Move all masks to specified device."""
        return TokenMasks(
            text_mask=self.text_mask.to(device),
            nontext_mask=self.nontext_mask.to(device),
            output_mask=self.output_mask.to(device) if self.output_mask is not None else None,
            system_mask=self.system_mask.to(device) if self.system_mask is not None else None,
        )


@dataclass
class AttentionAnalysisResult:
    """Complete result of attention analysis."""
    metrics: Dict[str, ModalityMetrics]  # 'early', 'middle', 'late'
    token_masks: TokenMasks
    attention_heatmap: Optional[np.ndarray] = None
    layer_metrics: Optional[List[ModalityMetrics]] = None

    def summary(self) -> str:
        """Generate a summary string of the analysis."""
        lines = ["=" * 60]
        lines.append("Cross-Modal Attention Analysis Results")
        lines.append("=" * 60)
        lines.append(f"Total tokens: {self.token_masks.total_tokens}")
        lines.append(f"Text tokens: {self.token_masks.num_text_tokens}")
        lines.append(f"Non-text tokens: {self.token_masks.num_nontext_tokens}")
        lines.append("-" * 60)

        for stage in ['early', 'middle', 'late']:
            if stage in self.metrics:
                m = self.metrics[stage]
                lines.append(f"{stage.upper()} layers:")
                lines.append(f"  MDI: {m.mdi:.2f}")
                lines.append(f"  AEI (text): {m.aei_text:.2f}")
                lines.append(f"  Text attention: {m.text_attention:.4f}")

        lines.append("=" * 60)
        return "\n".join(lines)


class AttentionAnalyzer:
    """
    Analyzer for cross-modal attention in MLLMs.

    Computes MDI and AEI metrics and provides visualization utilities.
    """

    def __init__(self, eps: float = 1e-10):
        """
        Initialize analyzer.

        Args:
            eps: Small value to prevent division by zero
        """
        self.eps = eps
        self.mdi_calc = MDI(eps=eps)
        self.aei_calc = AEI(eps=eps)

    def create_token_masks(
        self,
        total_length: int,
        text_start: int,
        text_end: int,
        nontext_start: int,
        nontext_end: int,
        output_start: Optional[int] = None,
        device: torch.device = torch.device('cpu'),
    ) -> TokenMasks:
        """
        Create token masks for different modalities.

        Args:
            total_length: Total sequence length
            text_start: Start index of text tokens
            text_end: End index of text tokens (exclusive)
            nontext_start: Start index of non-text tokens
            nontext_end: End index of non-text tokens (exclusive)
            output_start: Start index of output tokens (optional)
            device: Device to create tensors on

        Returns:
            TokenMasks object
        """
        text_mask = torch.zeros(total_length, dtype=torch.bool, device=device)
        nontext_mask = torch.zeros(total_length, dtype=torch.bool, device=device)

        text_mask[text_start:text_end] = True
        nontext_mask[nontext_start:nontext_end] = True

        output_mask = None
        if output_start is not None:
            output_mask = torch.zeros(total_length, dtype=torch.bool, device=device)
            output_mask[output_start:] = True

        return TokenMasks(
            text_mask=text_mask,
            nontext_mask=nontext_mask,
            output_mask=output_mask,
        )

    def create_masks_from_input_ids(
        self,
        input_ids: torch.Tensor,
        image_token_id: int,
        text_token_ids: Optional[List[int]] = None,
        special_token_ids: Optional[List[int]] = None,
    ) -> TokenMasks:
        """
        Create token masks from input_ids tensor.

        Args:
            input_ids: Token IDs tensor (batch, seq_len) or (seq_len,)
            image_token_id: Token ID used for image/non-text placeholders
            text_token_ids: Optional list of text-only token IDs
            special_token_ids: Token IDs to exclude (e.g., padding, special tokens)

        Returns:
            TokenMasks object
        """
        if input_ids.dim() > 1:
            input_ids = input_ids[0]  # Take first batch

        seq_len = len(input_ids)
        device = input_ids.device

        # Non-text tokens are those matching image_token_id
        nontext_mask = (input_ids == image_token_id)

        # Text tokens are everything else (excluding special tokens if specified)
        if special_token_ids:
            special_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
            for token_id in special_token_ids:
                special_mask |= (input_ids == token_id)
            text_mask = ~nontext_mask & ~special_mask
        else:
            text_mask = ~nontext_mask

        return TokenMasks(
            text_mask=text_mask,
            nontext_mask=nontext_mask,
        )

    def analyze(
        self,
        attention_weights: Union[torch.Tensor, List[torch.Tensor]],
        token_masks: TokenMasks,
        output_token_indices: Optional[torch.Tensor] = None,
    ) -> AttentionAnalysisResult:
        """
        Perform complete attention analysis.

        Args:
            attention_weights: Attention tensor(s) - single or list per layer
            token_masks: TokenMasks object with modality masks
            output_token_indices: Optional indices of output tokens

        Returns:
            AttentionAnalysisResult with metrics and analysis data
        """
        # Move masks to same device as attention
        if isinstance(attention_weights, list):
            device = attention_weights[0].device
        else:
            device = attention_weights.device
        token_masks = token_masks.to(device)

        # Compute layerwise metrics if list provided
        if isinstance(attention_weights, list):
            metrics_dict = compute_modality_metrics(
                attention_weights,
                token_masks.text_mask,
                token_masks.nontext_mask,
                output_token_indices,
                layerwise=True,
            )

            # Extract layer-by-layer metrics
            layer_metrics = metrics_dict.pop('all_layers', None)

            # Compute attention heatmap from middle layer
            if len(attention_weights) > 2:
                mid_idx = len(attention_weights) // 2
                heatmap = self._compute_heatmap(
                    attention_weights[mid_idx],
                    token_masks,
                )
            else:
                heatmap = None

        else:
            metrics_dict = {
                'all': compute_modality_metrics(
                    attention_weights,
                    token_masks.text_mask,
                    token_masks.nontext_mask,
                    output_token_indices,
                )
            }
            layer_metrics = None
            heatmap = self._compute_heatmap(attention_weights, token_masks)

        return AttentionAnalysisResult(
            metrics=metrics_dict,
            token_masks=token_masks,
            attention_heatmap=heatmap,
            layer_metrics=layer_metrics,
        )

    def _compute_heatmap(
        self,
        attention_weights: torch.Tensor,
        token_masks: TokenMasks,
    ) -> np.ndarray:
        """Compute aggregated attention heatmap."""
        # Average over batch and heads
        if attention_weights.dim() == 4:
            attn = attention_weights.mean(dim=(0, 1))
        elif attention_weights.dim() == 3:
            attn = attention_weights.mean(dim=0)
        else:
            attn = attention_weights

        return attn.cpu().numpy()

    def compute_attention_flow(
        self,
        attention_weights: List[torch.Tensor],
        token_masks: TokenMasks,
    ) -> Dict[str, List[float]]:
        """
        Compute attention flow across layers.

        Tracks how attention to text vs non-text changes through the network.

        Args:
            attention_weights: List of attention tensors per layer
            token_masks: TokenMasks object

        Returns:
            Dictionary with lists of attention proportions per layer
        """
        text_flow = []
        nontext_flow = []
        mdi_flow = []

        for attn in attention_weights:
            result = self.mdi_calc.compute(
                attn,
                token_masks.text_mask,
                token_masks.nontext_mask,
            )
            text_flow.append(result.text_attention)
            nontext_flow.append(result.nontext_attention)
            mdi_flow.append(result.mdi)

        return {
            'text_attention': text_flow,
            'nontext_attention': nontext_flow,
            'mdi': mdi_flow,
            'layer_indices': list(range(len(attention_weights))),
        }


def analyze_cross_modal_attention(
    attention_weights: Union[torch.Tensor, List[torch.Tensor]],
    text_token_mask: torch.Tensor,
    nontext_token_mask: torch.Tensor,
    output_token_indices: Optional[torch.Tensor] = None,
) -> AttentionAnalysisResult:
    """
    Convenience function for cross-modal attention analysis.

    Args:
        attention_weights: Attention tensor(s)
        text_token_mask: Boolean mask for text tokens
        nontext_token_mask: Boolean mask for non-text tokens
        output_token_indices: Optional indices of output tokens

    Returns:
        AttentionAnalysisResult
    """
    analyzer = AttentionAnalyzer()
    token_masks = TokenMasks(
        text_mask=text_token_mask,
        nontext_mask=nontext_token_mask,
    )
    return analyzer.analyze(attention_weights, token_masks, output_token_indices)
