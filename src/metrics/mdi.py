"""
Modality Dominance Index (MDI) Implementation

MDI quantifies the relative reliance of a multimodal model on textual versus
non-textual inputs during autoregressive generation.

MDI = (A_T / |T|) / (A_O / |O|)

Where:
- A_T: Total attention score for text tokens
- A_O: Total attention score for non-text tokens
- |T|: Number of text tokens
- |O|: Number of non-text tokens

MDI > 1: Text dominance
MDI < 1: Non-text dominance
MDI ≈ 1: Balanced attention
"""

import torch
import numpy as np
from typing import Union, Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MDIResult:
    """Result container for MDI computation."""
    mdi: float
    text_attention: float
    nontext_attention: float
    text_token_count: int
    nontext_token_count: int
    per_token_text_attention: float
    per_token_nontext_attention: float

    def __repr__(self) -> str:
        return (
            f"MDIResult(mdi={self.mdi:.4f}, "
            f"text_attn={self.text_attention:.4f}, "
            f"nontext_attn={self.nontext_attention:.4f})"
        )


class MDI:
    """
    Modality Dominance Index calculator.

    Measures the relative attention allocation between text and non-text modalities
    on a per-token basis during generation.
    """

    def __init__(self, eps: float = 1e-10):
        """
        Initialize MDI calculator.

        Args:
            eps: Small epsilon value to prevent division by zero
        """
        self.eps = eps

    def compute(
        self,
        attention_weights: torch.Tensor,
        text_token_mask: torch.Tensor,
        nontext_token_mask: torch.Tensor,
        output_token_indices: Optional[torch.Tensor] = None,
    ) -> MDIResult:
        """
        Compute MDI from attention weights.

        Args:
            attention_weights: Attention matrix of shape (batch, heads, seq_len, seq_len)
                              or (batch, seq_len, seq_len) if already averaged over heads
            text_token_mask: Boolean mask indicating text token positions (seq_len,)
            nontext_token_mask: Boolean mask indicating non-text token positions (seq_len,)
            output_token_indices: Optional indices of generated output tokens to analyze.
                                 If None, uses all positions after input tokens.

        Returns:
            MDIResult containing MDI value and component metrics
        """
        # Ensure tensors are on same device
        device = attention_weights.device
        text_token_mask = text_token_mask.to(device)
        nontext_token_mask = nontext_token_mask.to(device)

        # Average over heads if needed
        if attention_weights.dim() == 4:
            # (batch, heads, seq_len, seq_len) -> (batch, seq_len, seq_len)
            attention_weights = attention_weights.mean(dim=1)

        # Average over batch if needed
        if attention_weights.dim() == 3:
            attention_weights = attention_weights.mean(dim=0)

        # attention_weights is now (seq_len, seq_len)
        seq_len = attention_weights.shape[0]

        # Determine output token positions
        if output_token_indices is None:
            # Assume output tokens are positions where neither text nor nontext mask is True
            input_mask = text_token_mask | nontext_token_mask
            output_token_indices = torch.where(~input_mask)[0]

            # If no clear output tokens, use last portion of sequence
            if len(output_token_indices) == 0:
                # Use attention from all positions to input tokens
                output_token_indices = torch.arange(seq_len, device=device)

        # Extract attention from output tokens to input tokens
        # Shape: (num_output_tokens, seq_len)
        output_attention = attention_weights[output_token_indices]

        # Compute total attention to text and non-text tokens
        # Sum attention across all output tokens, then sum across respective input positions
        text_attention = output_attention[:, text_token_mask].sum().item()
        nontext_attention = output_attention[:, nontext_token_mask].sum().item()

        # Get token counts
        text_token_count = text_token_mask.sum().item()
        nontext_token_count = nontext_token_mask.sum().item()

        # Normalize to ensure A_T + A_O = 1 (as per paper)
        total_attention = text_attention + nontext_attention + self.eps
        text_attention_norm = text_attention / total_attention
        nontext_attention_norm = nontext_attention / total_attention

        # Compute per-token attention
        per_token_text = text_attention_norm / (text_token_count + self.eps)
        per_token_nontext = nontext_attention_norm / (nontext_token_count + self.eps)

        # Compute MDI: ratio of per-token attention
        mdi = per_token_text / (per_token_nontext + self.eps)

        return MDIResult(
            mdi=mdi,
            text_attention=text_attention_norm,
            nontext_attention=nontext_attention_norm,
            text_token_count=int(text_token_count),
            nontext_token_count=int(nontext_token_count),
            per_token_text_attention=per_token_text,
            per_token_nontext_attention=per_token_nontext,
        )

    def compute_layerwise(
        self,
        attention_weights_per_layer: List[torch.Tensor],
        text_token_mask: torch.Tensor,
        nontext_token_mask: torch.Tensor,
        output_token_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, List[MDIResult]]:
        """
        Compute MDI for each layer and aggregate by layer position.

        Args:
            attention_weights_per_layer: List of attention matrices, one per layer
            text_token_mask: Boolean mask for text tokens
            nontext_token_mask: Boolean mask for non-text tokens
            output_token_indices: Optional indices of output tokens

        Returns:
            Dictionary with 'early', 'middle', 'late' keys containing MDI results
        """
        num_layers = len(attention_weights_per_layer)

        # Compute MDI for each layer
        layer_results = []
        for attn in attention_weights_per_layer:
            result = self.compute(
                attn, text_token_mask, nontext_token_mask, output_token_indices
            )
            layer_results.append(result)

        # Aggregate by layer position (as per paper: first 2, middle 2, last 2)
        early_layers = layer_results[:2]
        late_layers = layer_results[-2:]

        if num_layers > 4:
            mid_start = num_layers // 2 - 1
            middle_layers = layer_results[mid_start:mid_start + 2]
        else:
            middle_layers = layer_results[1:3] if num_layers >= 3 else layer_results

        return {
            'early': early_layers,
            'middle': middle_layers,
            'late': late_layers,
            'all': layer_results,
        }

    @staticmethod
    def aggregate_results(results: List[MDIResult]) -> MDIResult:
        """Average multiple MDI results."""
        if not results:
            raise ValueError("Cannot aggregate empty results list")

        avg_mdi = np.mean([r.mdi for r in results])
        avg_text_attn = np.mean([r.text_attention for r in results])
        avg_nontext_attn = np.mean([r.nontext_attention for r in results])
        avg_text_count = int(np.mean([r.text_token_count for r in results]))
        avg_nontext_count = int(np.mean([r.nontext_token_count for r in results]))
        avg_per_token_text = np.mean([r.per_token_text_attention for r in results])
        avg_per_token_nontext = np.mean([r.per_token_nontext_attention for r in results])

        return MDIResult(
            mdi=avg_mdi,
            text_attention=avg_text_attn,
            nontext_attention=avg_nontext_attn,
            text_token_count=avg_text_count,
            nontext_token_count=avg_nontext_count,
            per_token_text_attention=avg_per_token_text,
            per_token_nontext_attention=avg_per_token_nontext,
        )


def compute_mdi(
    attention_weights: Union[torch.Tensor, np.ndarray],
    text_token_mask: Union[torch.Tensor, np.ndarray],
    nontext_token_mask: Union[torch.Tensor, np.ndarray],
    eps: float = 1e-10,
) -> float:
    """
    Convenience function to compute MDI value directly.

    Args:
        attention_weights: Attention matrix
        text_token_mask: Boolean mask for text tokens
        nontext_token_mask: Boolean mask for non-text tokens
        eps: Small value to prevent division by zero

    Returns:
        MDI value (float)
    """
    # Convert numpy arrays to tensors if needed
    if isinstance(attention_weights, np.ndarray):
        attention_weights = torch.from_numpy(attention_weights)
    if isinstance(text_token_mask, np.ndarray):
        text_token_mask = torch.from_numpy(text_token_mask)
    if isinstance(nontext_token_mask, np.ndarray):
        nontext_token_mask = torch.from_numpy(nontext_token_mask)

    calculator = MDI(eps=eps)
    result = calculator.compute(attention_weights, text_token_mask, nontext_token_mask)
    return result.mdi
