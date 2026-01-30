"""
[CLS]-based Token Pruning for Visual Token Compression.

Implements the token compression strategy from the paper:
"The [CLS] token is designed to capture the global semantics of
the image via self-attention and provides stable visual token
saliency assessments consistent across network layers."

Key equations from paper:
- Importance score: s_i = Attn([CLS], v_i)
- Retained tokens: M = N(1 - r), where r is reduction rate
- Adaptive threshold: τ = min{τ | |{a ∈ a_[CLS] | a ≥ τ}| ≤ N × (1 - R)}
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np


@dataclass
class PruningResult:
    """Result container for token pruning."""
    pruned_tokens: torch.Tensor
    kept_indices: torch.Tensor
    pruned_indices: torch.Tensor
    importance_scores: torch.Tensor
    reduction_rate: float
    original_count: int
    retained_count: int

    def __repr__(self) -> str:
        return (
            f"PruningResult(reduction={self.reduction_rate:.2%}, "
            f"retained={self.retained_count}/{self.original_count})"
        )


def compute_cls_attention_scores(
    attention_weights: torch.Tensor,
    cls_token_index: int = 0,
) -> torch.Tensor:
    """
    Compute importance scores for visual tokens using [CLS] attention.

    From the paper:
    "Given N visual tokens V = {v_1, ..., v_N} encoded by a visual transformer,
    we compute the importance score s_i for each token v_i as:
    s_i = Attn([CLS], v_i)"

    Args:
        attention_weights: Attention matrix from visual encoder
                          Shape: (batch, heads, seq, seq) or (heads, seq, seq)
        cls_token_index: Index of [CLS] token (usually 0)

    Returns:
        Importance scores for each token, shape: (seq_len - 1,) excluding CLS
    """
    # Handle different input shapes
    if attention_weights.dim() == 4:
        # (batch, heads, seq, seq) -> average over batch and heads
        attn = attention_weights.mean(dim=(0, 1))
    elif attention_weights.dim() == 3:
        # (heads, seq, seq) -> average over heads
        attn = attention_weights.mean(dim=0)
    else:
        attn = attention_weights

    # Get attention from [CLS] token to all other tokens
    # Shape: (seq_len,)
    cls_attention = attn[cls_token_index]

    # Exclude [CLS] token itself
    # Return attention to visual tokens (excluding CLS)
    visual_token_scores = torch.cat([
        cls_attention[:cls_token_index],
        cls_attention[cls_token_index + 1:]
    ])

    return visual_token_scores


def compute_adaptive_threshold(
    importance_scores: torch.Tensor,
    reduction_rate: float,
) -> float:
    """
    Compute adaptive threshold for token pruning.

    From paper equation (8):
    τ = min{τ | |{a ∈ a_[CLS] | a ≥ τ}| ≤ N × (1 - R)}

    Args:
        importance_scores: Importance scores for each token
        reduction_rate: Target reduction rate R (e.g., 0.75 for 75% reduction)

    Returns:
        Threshold value τ
    """
    N = len(importance_scores)
    target_count = int(N * (1 - reduction_rate))

    if target_count <= 0:
        return importance_scores.max().item() + 1.0
    if target_count >= N:
        return importance_scores.min().item() - 1.0

    # Sort scores in descending order
    sorted_scores, _ = torch.sort(importance_scores, descending=True)

    # Find threshold: score at position target_count
    threshold = sorted_scores[target_count - 1].item()

    return threshold


def prune_visual_tokens(
    visual_tokens: torch.Tensor,
    importance_scores: torch.Tensor,
    reduction_rate: float = 0.75,
    method: str = "topk",
) -> PruningResult:
    """
    Prune visual tokens based on importance scores.

    From paper:
    "Then, applying a token reduction rate r, only the top M = N(1 - r)
    tokens with the highest scores are retained, forming a compressed
    sequence V' = {v'_1, ..., v'_M}."

    Args:
        visual_tokens: Visual token embeddings, shape: (batch, N, dim) or (N, dim)
        importance_scores: Importance scores for each token
        reduction_rate: Fraction of tokens to remove (0.75 = keep 25%)
        method: Pruning method ('topk' or 'threshold')

    Returns:
        PruningResult with pruned tokens and metadata
    """
    has_batch = visual_tokens.dim() == 3
    if not has_batch:
        visual_tokens = visual_tokens.unsqueeze(0)

    batch_size, N, dim = visual_tokens.shape
    M = int(N * (1 - reduction_rate))  # Number of tokens to keep
    M = max(1, M)  # Keep at least 1 token

    if method == "topk":
        # Keep top-M tokens with highest importance scores
        _, kept_indices = torch.topk(importance_scores, M, largest=True, sorted=False)
        kept_indices, _ = torch.sort(kept_indices)  # Maintain order

    elif method == "threshold":
        # Use adaptive threshold
        threshold = compute_adaptive_threshold(importance_scores, reduction_rate)
        kept_indices = torch.where(importance_scores >= threshold)[0]

        # Ensure we don't keep more than target
        if len(kept_indices) > M:
            _, topk_indices = torch.topk(
                importance_scores[kept_indices], M, largest=True
            )
            kept_indices = kept_indices[topk_indices]
        elif len(kept_indices) == 0:
            # Fallback to topk if threshold removes all
            _, kept_indices = torch.topk(importance_scores, M, largest=True)

        kept_indices, _ = torch.sort(kept_indices)

    else:
        raise ValueError(f"Unknown pruning method: {method}")

    # Get pruned token indices
    all_indices = torch.arange(N, device=visual_tokens.device)
    pruned_mask = ~torch.isin(all_indices, kept_indices)
    pruned_indices = all_indices[pruned_mask]

    # Extract kept tokens
    pruned_tokens = visual_tokens[:, kept_indices, :]

    if not has_batch:
        pruned_tokens = pruned_tokens.squeeze(0)

    return PruningResult(
        pruned_tokens=pruned_tokens,
        kept_indices=kept_indices,
        pruned_indices=pruned_indices,
        importance_scores=importance_scores,
        reduction_rate=reduction_rate,
        original_count=N,
        retained_count=len(kept_indices),
    )


class CLSTokenPruner:
    """
    [CLS]-based token pruner for visual tokens.

    Implements the FasterVLM-style token compression to mitigate
    text dominance in MLLMs.
    """

    def __init__(
        self,
        reduction_rate: float = 0.75,
        cls_token_index: int = 0,
        method: str = "topk",
    ):
        """
        Initialize the pruner.

        Args:
            reduction_rate: Fraction of tokens to remove (0.75, 0.90, 0.95)
            cls_token_index: Index of [CLS] token in attention matrix
            method: Pruning method ('topk' or 'threshold')
        """
        self.reduction_rate = reduction_rate
        self.cls_token_index = cls_token_index
        self.method = method

    def compute_scores(
        self,
        attention_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Compute importance scores from attention weights."""
        return compute_cls_attention_scores(
            attention_weights,
            self.cls_token_index,
        )

    def prune(
        self,
        visual_tokens: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> PruningResult:
        """
        Prune visual tokens using [CLS] attention.

        Args:
            visual_tokens: Visual token embeddings
            attention_weights: Attention weights from visual encoder

        Returns:
            PruningResult with compressed tokens
        """
        scores = self.compute_scores(attention_weights)
        return prune_visual_tokens(
            visual_tokens,
            scores,
            self.reduction_rate,
            self.method,
        )

    def __call__(
        self,
        visual_tokens: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> PruningResult:
        """Callable interface for pruning."""
        return self.prune(visual_tokens, attention_weights)


class FasterVLM(nn.Module):
    """
    FasterVLM: Token compression module for vision-language models.

    Applies [CLS]-guided token pruning to reduce redundant visual tokens
    before fusion with the LLM, effectively mitigating text dominance.

    From paper Table 2:
    - 75% reduction: MDI drops from 17.37 to 3.39
    - 90% reduction: MDI drops from 17.37 to 1.84
    - 95% reduction: MDI drops from 17.37 to 3.39 (middle: 0.86)
    """

    def __init__(
        self,
        reduction_rate: float = 0.90,
        cls_token_index: int = 0,
        learnable: bool = False,
    ):
        """
        Initialize FasterVLM module.

        Args:
            reduction_rate: Target token reduction rate
            cls_token_index: Index of [CLS] token
            learnable: Whether to use learnable importance weights
        """
        super().__init__()
        self.reduction_rate = reduction_rate
        self.cls_token_index = cls_token_index
        self.learnable = learnable

        if learnable:
            # Learnable temperature for softmax
            self.temperature = nn.Parameter(torch.ones(1))

    def forward(
        self,
        visual_tokens: torch.Tensor,
        attention_weights: torch.Tensor,
        return_indices: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply token compression.

        Args:
            visual_tokens: Input visual tokens (batch, N, dim)
            attention_weights: Attention from visual encoder (batch, heads, N+1, N+1)
            return_indices: Whether to return kept indices

        Returns:
            Compressed visual tokens, optionally with kept indices
        """
        # Compute importance scores
        scores = compute_cls_attention_scores(
            attention_weights,
            self.cls_token_index,
        )

        if self.learnable:
            scores = scores / self.temperature

        # Prune tokens
        result = prune_visual_tokens(
            visual_tokens,
            scores,
            self.reduction_rate,
            method="topk",
        )

        if return_indices:
            return result.pruned_tokens, result.kept_indices
        return result.pruned_tokens

    def set_reduction_rate(self, rate: float):
        """Update reduction rate (e.g., for progressive pruning)."""
        assert 0.0 <= rate < 1.0, "Reduction rate must be in [0, 1)"
        self.reduction_rate = rate


class ProgressiveTokenPruner:
    """
    Progressive token pruning across layers.

    Gradually increases compression through the network.
    """

    def __init__(
        self,
        initial_rate: float = 0.5,
        final_rate: float = 0.9,
        num_stages: int = 4,
    ):
        """
        Initialize progressive pruner.

        Args:
            initial_rate: Starting reduction rate
            final_rate: Final reduction rate
            num_stages: Number of pruning stages
        """
        self.rates = np.linspace(initial_rate, final_rate, num_stages)
        self.pruners = [CLSTokenPruner(rate) for rate in self.rates]

    def prune_stage(
        self,
        visual_tokens: torch.Tensor,
        attention_weights: torch.Tensor,
        stage: int,
    ) -> PruningResult:
        """Apply pruning at a specific stage."""
        if stage >= len(self.pruners):
            stage = len(self.pruners) - 1
        return self.pruners[stage](visual_tokens, attention_weights)


def apply_token_compression_to_llava(
    model,
    visual_tokens: torch.Tensor,
    reduction_rate: float = 0.90,
) -> torch.Tensor:
    """
    Apply token compression to LLaVA-style models.

    This is a utility function that extracts attention from the
    vision encoder and applies compression before LLM fusion.

    Args:
        model: LLaVA model with vision_tower
        visual_tokens: Visual tokens from vision encoder
        reduction_rate: Compression rate

    Returns:
        Compressed visual tokens
    """
    # Get attention from vision encoder
    vision_encoder = model.vision_tower if hasattr(model, 'vision_tower') else None
    if vision_encoder is None:
        return visual_tokens

    # Note: This requires the vision encoder to output attention
    # Most ViT models can be configured to do this

    pruner = CLSTokenPruner(reduction_rate=reduction_rate)

    # For actual implementation, you need to:
    # 1. Get attention from the last layer of vision encoder
    # 2. Apply pruning
    # 3. Return compressed tokens

    # Placeholder - actual attention extraction depends on model architecture
    return visual_tokens
