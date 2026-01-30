"""
Token compression methods for mitigating text dominance.

Implements [CLS]-based token pruning as described in the paper
to rebalance attention between text and non-text modalities.
"""

from .token_pruning import (
    CLSTokenPruner,
    compute_cls_attention_scores,
    prune_visual_tokens,
    FasterVLM,
)

__all__ = [
    "CLSTokenPruner",
    "compute_cls_attention_scores",
    "prune_visual_tokens",
    "FasterVLM",
]
