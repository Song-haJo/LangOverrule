"""
Combined metrics computation for MDI and AEI.
"""

import torch
import numpy as np
from typing import Union, Dict, List, Optional, NamedTuple
from dataclasses import dataclass

from .mdi import MDI, MDIResult
from .aei import AEI, AEIResult


@dataclass
class ModalityMetrics:
    """Combined container for both MDI and AEI metrics."""
    mdi: float
    aei_text: float
    aei_nontext: float
    text_attention: float
    nontext_attention: float
    text_token_count: int
    nontext_token_count: int
    layer_stage: str = "all"  # 'early', 'middle', 'late', or 'all'

    def __repr__(self) -> str:
        return (
            f"ModalityMetrics({self.layer_stage}: "
            f"MDI={self.mdi:.2f}, AEI_text={self.aei_text:.2f})"
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary for easy serialization."""
        return {
            'mdi': self.mdi,
            'aei_text': self.aei_text,
            'aei_nontext': self.aei_nontext,
            'text_attention': self.text_attention,
            'nontext_attention': self.nontext_attention,
            'text_token_count': self.text_token_count,
            'nontext_token_count': self.nontext_token_count,
            'layer_stage': self.layer_stage,
        }


class ModalityMetricsCalculator:
    """
    Calculator for combined MDI and AEI metrics.

    This class provides a unified interface for computing both metrics
    and organizing results by layer stages as described in the paper.
    """

    def __init__(self, eps: float = 1e-10):
        self.mdi_calc = MDI(eps=eps)
        self.aei_calc = AEI(eps=eps)
        self.eps = eps

    def compute(
        self,
        attention_weights: torch.Tensor,
        text_token_mask: torch.Tensor,
        nontext_token_mask: torch.Tensor,
        output_token_indices: Optional[torch.Tensor] = None,
    ) -> ModalityMetrics:
        """
        Compute combined MDI and AEI metrics.

        Args:
            attention_weights: Attention matrix
            text_token_mask: Boolean mask for text tokens
            nontext_token_mask: Boolean mask for non-text tokens
            output_token_indices: Optional indices of output tokens

        Returns:
            ModalityMetrics containing both MDI and AEI values
        """
        mdi_result = self.mdi_calc.compute(
            attention_weights, text_token_mask, nontext_token_mask, output_token_indices
        )
        aei_result = self.aei_calc.compute(
            attention_weights, text_token_mask, nontext_token_mask, output_token_indices
        )

        return ModalityMetrics(
            mdi=mdi_result.mdi,
            aei_text=aei_result.aei_text,
            aei_nontext=aei_result.aei_nontext,
            text_attention=mdi_result.text_attention,
            nontext_attention=mdi_result.nontext_attention,
            text_token_count=mdi_result.text_token_count,
            nontext_token_count=mdi_result.nontext_token_count,
        )

    def compute_layerwise(
        self,
        attention_weights_per_layer: List[torch.Tensor],
        text_token_mask: torch.Tensor,
        nontext_token_mask: torch.Tensor,
        output_token_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, ModalityMetrics]:
        """
        Compute metrics for each layer stage (early, middle, late).

        Following the paper's methodology:
        - Early: First 2 layers
        - Middle: Middle 2 layers
        - Late: Last 2 layers

        Args:
            attention_weights_per_layer: List of attention matrices per layer
            text_token_mask: Boolean mask for text tokens
            nontext_token_mask: Boolean mask for non-text tokens
            output_token_indices: Optional indices of output tokens

        Returns:
            Dictionary with 'early', 'middle', 'late' keys containing ModalityMetrics
        """
        num_layers = len(attention_weights_per_layer)

        # Compute metrics for each layer
        all_metrics = []
        for attn in attention_weights_per_layer:
            metrics = self.compute(
                attn, text_token_mask, nontext_token_mask, output_token_indices
            )
            all_metrics.append(metrics)

        # Define layer ranges
        early_indices = list(range(min(2, num_layers)))
        late_indices = list(range(max(0, num_layers - 2), num_layers))

        if num_layers > 4:
            mid_start = num_layers // 2 - 1
            middle_indices = list(range(mid_start, min(mid_start + 2, num_layers)))
        else:
            middle_indices = list(range(1, min(3, num_layers)))

        def aggregate_metrics(indices: List[int], stage: str) -> ModalityMetrics:
            """Aggregate metrics from specified layer indices."""
            metrics_list = [all_metrics[i] for i in indices if i < len(all_metrics)]
            if not metrics_list:
                return all_metrics[0] if all_metrics else None

            return ModalityMetrics(
                mdi=np.mean([m.mdi for m in metrics_list]),
                aei_text=np.mean([m.aei_text for m in metrics_list]),
                aei_nontext=np.mean([m.aei_nontext for m in metrics_list]),
                text_attention=np.mean([m.text_attention for m in metrics_list]),
                nontext_attention=np.mean([m.nontext_attention for m in metrics_list]),
                text_token_count=int(np.mean([m.text_token_count for m in metrics_list])),
                nontext_token_count=int(np.mean([m.nontext_token_count for m in metrics_list])),
                layer_stage=stage,
            )

        return {
            'early': aggregate_metrics(early_indices, 'early'),
            'middle': aggregate_metrics(middle_indices, 'middle'),
            'late': aggregate_metrics(late_indices, 'late'),
            'all_layers': all_metrics,
        }


def compute_modality_metrics(
    attention_weights: Union[torch.Tensor, List[torch.Tensor]],
    text_token_mask: Union[torch.Tensor, np.ndarray],
    nontext_token_mask: Union[torch.Tensor, np.ndarray],
    output_token_indices: Optional[torch.Tensor] = None,
    layerwise: bool = False,
    eps: float = 1e-10,
) -> Union[ModalityMetrics, Dict[str, ModalityMetrics]]:
    """
    Convenience function to compute modality metrics.

    Args:
        attention_weights: Single attention matrix or list of matrices per layer
        text_token_mask: Boolean mask for text tokens
        nontext_token_mask: Boolean mask for non-text tokens
        output_token_indices: Optional indices of output tokens
        layerwise: If True and attention_weights is a list, compute layerwise metrics
        eps: Small value to prevent division by zero

    Returns:
        ModalityMetrics or Dict of ModalityMetrics if layerwise=True
    """
    # Convert numpy arrays if needed
    if isinstance(text_token_mask, np.ndarray):
        text_token_mask = torch.from_numpy(text_token_mask)
    if isinstance(nontext_token_mask, np.ndarray):
        nontext_token_mask = torch.from_numpy(nontext_token_mask)

    calculator = ModalityMetricsCalculator(eps=eps)

    if isinstance(attention_weights, list) and layerwise:
        return calculator.compute_layerwise(
            attention_weights, text_token_mask, nontext_token_mask, output_token_indices
        )
    else:
        if isinstance(attention_weights, list):
            # Average across layers
            attention_weights = torch.stack(attention_weights).mean(dim=0)
        return calculator.compute(
            attention_weights, text_token_mask, nontext_token_mask, output_token_indices
        )


def format_metrics_table(
    metrics_dict: Dict[str, ModalityMetrics],
    model_name: str = "Model"
) -> str:
    """
    Format metrics as a table string (similar to Table 1 in the paper).

    Args:
        metrics_dict: Dictionary with 'early', 'middle', 'late' keys
        model_name: Name of the model for display

    Returns:
        Formatted table string
    """
    header = f"{'Model':<30} {'Stage':<10} {'MDI':>10} {'AEI':>10}"
    separator = "-" * 65

    lines = [header, separator]

    for stage in ['early', 'middle', 'late']:
        if stage in metrics_dict and metrics_dict[stage] is not None:
            m = metrics_dict[stage]
            line = f"{model_name:<30} {stage.capitalize():<10} {m.mdi:>10.2f} {m.aei_text:>10.2f}"
            lines.append(line)

    return "\n".join(lines)
