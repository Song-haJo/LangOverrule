"""
Attention Efficiency Index (AEI) Implementation

AEI measures the efficiency of a modality in converting its token representation
into attention, providing a normalized assessment of resource usage.

AEI_T = P_T / Q_T

Where:
- P_T = A_T / (A_T + A_O) : Proportion of attention captured by text modality
- Q_T = |T| / (|T| + |O|) : Proportional size of text modality in input

AEI > 1: High efficiency (modality gets more attention than its token share)
AEI < 1: Low efficiency (modality gets less attention than its token share)
AEI ≈ 1: Proportional attention allocation
"""

import torch
import numpy as np
from typing import Union, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class AEIResult:
    """Result container for AEI computation."""
    aei_text: float  # AEI for text modality
    aei_nontext: float  # AEI for non-text modality
    attention_proportion_text: float  # P_T
    attention_proportion_nontext: float  # P_O
    token_proportion_text: float  # Q_T
    token_proportion_nontext: float  # Q_O

    def __repr__(self) -> str:
        return (
            f"AEIResult(aei_text={self.aei_text:.4f}, "
            f"aei_nontext={self.aei_nontext:.4f})"
        )


class AEI:
    """
    Attention Efficiency Index calculator.

    Measures how efficiently each modality converts its token representation
    into attention during generation.
    """

    def __init__(self, eps: float = 1e-10):
        """
        Initialize AEI calculator.

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
    ) -> AEIResult:
        """
        Compute AEI from attention weights.

        Args:
            attention_weights: Attention matrix of shape (batch, heads, seq_len, seq_len)
                              or (batch, seq_len, seq_len) if already averaged over heads
            text_token_mask: Boolean mask indicating text token positions (seq_len,)
            nontext_token_mask: Boolean mask indicating non-text token positions (seq_len,)
            output_token_indices: Optional indices of generated output tokens to analyze

        Returns:
            AEIResult containing AEI values for both modalities
        """
        device = attention_weights.device
        text_token_mask = text_token_mask.to(device)
        nontext_token_mask = nontext_token_mask.to(device)

        # Average over heads if needed
        if attention_weights.dim() == 4:
            attention_weights = attention_weights.mean(dim=1)

        # Average over batch if needed
        if attention_weights.dim() == 3:
            attention_weights = attention_weights.mean(dim=0)

        seq_len = attention_weights.shape[0]

        # Determine output token positions
        if output_token_indices is None:
            input_mask = text_token_mask | nontext_token_mask
            output_token_indices = torch.where(~input_mask)[0]
            if len(output_token_indices) == 0:
                output_token_indices = torch.arange(seq_len, device=device)

        # Extract attention from output tokens
        output_attention = attention_weights[output_token_indices]

        # Compute attention to each modality
        text_attention = output_attention[:, text_token_mask].sum().item()
        nontext_attention = output_attention[:, nontext_token_mask].sum().item()
        total_attention = text_attention + nontext_attention + self.eps

        # Compute attention proportions (P_T and P_O)
        P_T = text_attention / total_attention
        P_O = nontext_attention / total_attention

        # Get token counts
        num_text = text_token_mask.sum().item()
        num_nontext = nontext_token_mask.sum().item()
        total_tokens = num_text + num_nontext + self.eps

        # Compute token proportions (Q_T and Q_O)
        Q_T = num_text / total_tokens
        Q_O = num_nontext / total_tokens

        # Compute AEI for each modality
        aei_text = P_T / (Q_T + self.eps)
        aei_nontext = P_O / (Q_O + self.eps)

        return AEIResult(
            aei_text=aei_text,
            aei_nontext=aei_nontext,
            attention_proportion_text=P_T,
            attention_proportion_nontext=P_O,
            token_proportion_text=Q_T,
            token_proportion_nontext=Q_O,
        )

    def compute_layerwise(
        self,
        attention_weights_per_layer: List[torch.Tensor],
        text_token_mask: torch.Tensor,
        nontext_token_mask: torch.Tensor,
        output_token_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, List[AEIResult]]:
        """
        Compute AEI for each layer and aggregate by layer position.

        Args:
            attention_weights_per_layer: List of attention matrices, one per layer
            text_token_mask: Boolean mask for text tokens
            nontext_token_mask: Boolean mask for non-text tokens
            output_token_indices: Optional indices of output tokens

        Returns:
            Dictionary with 'early', 'middle', 'late' keys containing AEI results
        """
        num_layers = len(attention_weights_per_layer)

        # Compute AEI for each layer
        layer_results = []
        for attn in attention_weights_per_layer:
            result = self.compute(
                attn, text_token_mask, nontext_token_mask, output_token_indices
            )
            layer_results.append(result)

        # Aggregate by layer position
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
    def aggregate_results(results: List[AEIResult]) -> AEIResult:
        """Average multiple AEI results."""
        if not results:
            raise ValueError("Cannot aggregate empty results list")

        return AEIResult(
            aei_text=np.mean([r.aei_text for r in results]),
            aei_nontext=np.mean([r.aei_nontext for r in results]),
            attention_proportion_text=np.mean([r.attention_proportion_text for r in results]),
            attention_proportion_nontext=np.mean([r.attention_proportion_nontext for r in results]),
            token_proportion_text=np.mean([r.token_proportion_text for r in results]),
            token_proportion_nontext=np.mean([r.token_proportion_nontext for r in results]),
        )


def compute_aei(
    attention_weights: Union[torch.Tensor, np.ndarray],
    text_token_mask: Union[torch.Tensor, np.ndarray],
    nontext_token_mask: Union[torch.Tensor, np.ndarray],
    eps: float = 1e-10,
) -> float:
    """
    Convenience function to compute text AEI value directly.

    Args:
        attention_weights: Attention matrix
        text_token_mask: Boolean mask for text tokens
        nontext_token_mask: Boolean mask for non-text tokens
        eps: Small value to prevent division by zero

    Returns:
        AEI value for text modality (float)
    """
    if isinstance(attention_weights, np.ndarray):
        attention_weights = torch.from_numpy(attention_weights)
    if isinstance(text_token_mask, np.ndarray):
        text_token_mask = torch.from_numpy(text_token_mask)
    if isinstance(nontext_token_mask, np.ndarray):
        nontext_token_mask = torch.from_numpy(nontext_token_mask)

    calculator = AEI(eps=eps)
    result = calculator.compute(attention_weights, text_token_mask, nontext_token_mask)
    return result.aei_text
