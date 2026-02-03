#!/usr/bin/env python3
"""
Simple test script for MDI (Modality Dominance Index) calculation.
"""

import torch
import numpy as np
from src.metrics.mdi import MDI, compute_mdi

def test_basic_mdi():
    """Test basic MDI computation with synthetic data."""
    print("="*60)
    print("Testing MDI Calculation")
    print("="*60)

    # Create synthetic attention weights
    # Simulating a scenario where model pays more attention to text tokens
    seq_len = 200
    batch_size = 1
    num_heads = 12

    # Create attention matrix (batch, heads, seq_len, seq_len)
    attention_weights = torch.randn(batch_size, num_heads, seq_len, seq_len)
    attention_weights = torch.softmax(attention_weights, dim=-1)

    # Define token masks
    # First 50 tokens: image tokens
    # Next 100 tokens: text tokens
    # Last 50 tokens: output tokens
    text_mask = torch.zeros(seq_len, dtype=torch.bool)
    text_mask[50:150] = True  # Text tokens at positions 50-150

    nontext_mask = torch.zeros(seq_len, dtype=torch.bool)
    nontext_mask[:50] = True  # Image tokens at positions 0-50

    # Calculate MDI
    calculator = MDI()
    result = calculator.compute(
        attention_weights,
        text_mask,
        nontext_mask
    )

    print(f"\nResults:")
    print(f"MDI: {result.mdi:.4f}")
    print(f"Text tokens: {result.text_token_count}")
    print(f"Non-text tokens: {result.nontext_token_count}")
    print(f"Text attention (normalized): {result.text_attention:.4f}")
    print(f"Non-text attention (normalized): {result.nontext_attention:.4f}")
    print(f"Per-token text attention: {result.per_token_text_attention:.4f}")
    print(f"Per-token non-text attention: {result.per_token_nontext_attention:.4f}")

    if result.mdi > 1:
        print(f"\n✓ Text dominance detected (MDI = {result.mdi:.2f} > 1)")
    elif result.mdi < 1:
        print(f"\n✓ Non-text dominance detected (MDI = {result.mdi:.2f} < 1)")
    else:
        print(f"\n✓ Balanced attention (MDI ≈ 1)")

    return result


def test_text_dominance_scenario():
    """Test scenario with strong text dominance."""
    print("\n" + "="*60)
    print("Testing Text Dominance Scenario")
    print("="*60)

    seq_len = 100

    # Create attention matrix where output tokens attend more to text
    attention_weights = torch.zeros(1, 1, seq_len, seq_len)

    # Text tokens: 30-70 (40 tokens)
    # Image tokens: 0-30 (30 tokens)
    # Output tokens: 70-100 (30 tokens)

    text_mask = torch.zeros(seq_len, dtype=torch.bool)
    text_mask[30:70] = True

    nontext_mask = torch.zeros(seq_len, dtype=torch.bool)
    nontext_mask[:30] = True

    # Make output tokens attend 80% to text, 20% to images
    for i in range(70, 100):
        attention_weights[0, 0, i, 30:70] = 0.8 / 40  # 80% to text tokens
        attention_weights[0, 0, i, :30] = 0.2 / 30     # 20% to image tokens

    calculator = MDI()
    result = calculator.compute(
        attention_weights,
        text_mask,
        nontext_mask
    )

    print(f"\nScenario: 80% attention to text, 20% to images")
    print(f"MDI: {result.mdi:.4f}")
    print(f"Expected behavior: MDI > 1 (text dominance)")

    return result


def test_balanced_scenario():
    """Test scenario with balanced attention."""
    print("\n" + "="*60)
    print("Testing Balanced Attention Scenario")
    print("="*60)

    seq_len = 100

    # Create attention matrix with balanced attention
    attention_weights = torch.zeros(1, 1, seq_len, seq_len)

    # Equal number of text and non-text tokens
    text_mask = torch.zeros(seq_len, dtype=torch.bool)
    text_mask[30:60] = True  # 30 text tokens

    nontext_mask = torch.zeros(seq_len, dtype=torch.bool)
    nontext_mask[:30] = True  # 30 image tokens

    # Make output tokens attend equally (per token basis)
    for i in range(60, 100):
        attention_weights[0, 0, i, 30:60] = 0.5 / 30  # 50% to text
        attention_weights[0, 0, i, :30] = 0.5 / 30     # 50% to images

    calculator = MDI()
    result = calculator.compute(
        attention_weights,
        text_mask,
        nontext_mask
    )

    print(f"\nScenario: Equal per-token attention to text and images")
    print(f"MDI: {result.mdi:.4f}")
    print(f"Expected behavior: MDI ≈ 1 (balanced)")

    return result


if __name__ == "__main__":
    # Run tests
    test_basic_mdi()
    test_text_dominance_scenario()
    test_balanced_scenario()

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
