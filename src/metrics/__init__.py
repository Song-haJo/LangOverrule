"""
Evaluation metrics for measuring text dominance in MLLMs.

Metrics:
- MDI (Modality Dominance Index): Measures relative attention per token between modalities
- AEI (Attention Efficiency Index): Measures attention efficiency relative to token proportion
"""

from .mdi import MDI, compute_mdi
from .aei import AEI, compute_aei
from .combined import compute_modality_metrics, ModalityMetrics

__all__ = [
    "MDI",
    "AEI",
    "compute_mdi",
    "compute_aei",
    "compute_modality_metrics",
    "ModalityMetrics",
]
