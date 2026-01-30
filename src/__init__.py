"""
Text Dominance Analysis in Multimodal Large Language Models

Re-implementation of "When Language Overrules: Revealing Text Dominance in MLLMs"
"""

__version__ = "0.1.0"
__author__ = "Reimplementation"

from .metrics import MDI, AEI, compute_modality_metrics
from .attention import AttentionExtractor, AttentionAnalyzer
from .compression import CLSTokenPruner

__all__ = [
    "MDI",
    "AEI",
    "compute_modality_metrics",
    "AttentionExtractor",
    "AttentionAnalyzer",
    "CLSTokenPruner",
]
