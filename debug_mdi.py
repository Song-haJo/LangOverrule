#!/usr/bin/env python3
"""Debug MDI calculation"""

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.metrics.mdi import MDI

# Create simple test case
# Attention matrix: 20x20 (20 tokens total)
# - 5 text tokens (indices 0-4)
# - 10 image tokens (indices 5-14)
# - 5 other tokens (indices 15-19)

torch.manual_seed(42)
attention = torch.softmax(torch.randn(20, 20), dim=-1)  # Random but normalized

# Create masks
text_mask = torch.zeros(20, dtype=torch.bool)
text_mask[0:5] = True

nontext_mask = torch.zeros(20, dtype=torch.bool)
nontext_mask[5:15] = True

# Query: use text tokens (0-4)
query_indices = torch.arange(5)

print("="*80)
print("MDI Calculation Debug")
print("="*80)
print(f"Attention shape: {attention.shape}")
print(f"Text tokens (query): {query_indices.tolist()}")
print(f"Text mask sum: {text_mask.sum()}")
print(f"Nontext mask sum: {nontext_mask.sum()}")

# Manual calculation
query_attention = attention[query_indices]  # Shape: (5, 20)
text_attention = query_attention[:, text_mask].sum().item()
nontext_attention = query_attention[:, nontext_mask].sum().item()

print(f"\nRaw attention sums:")
print(f"  Text attention: {text_attention:.6f}")
print(f"  Nontext attention: {nontext_attention:.6f}")

# Manual MDI
text_count = text_mask.sum().item()
nontext_count = nontext_mask.sum().item()
mdi_manual = (text_attention / text_count) / (nontext_attention / nontext_count)

print(f"\nManual MDI calculation:")
print(f"  Per-token text: {text_attention / text_count:.6f}")
print(f"  Per-token nontext: {nontext_attention / nontext_count:.6f}")
print(f"  MDI (manual): {mdi_manual:.6f}")

# Using MDI class
calculator = MDI()
result = calculator.compute(
    attention,
    text_mask,
    nontext_mask,
    output_token_indices=query_indices
)

print(f"\nMDI class calculation:")
print(f"  Text attention: {result.text_attention:.6f}")
print(f"  Nontext attention: {result.nontext_attention:.6f}")
print(f"  Per-token text: {result.per_token_text_attention:.6f}")
print(f"  Per-token nontext: {result.per_token_nontext_attention:.6f}")
print(f"  MDI (class): {result.mdi:.6f}")

print(f"\n{'='*80}")
print(f"Comparison: Manual={mdi_manual:.6f}, Class={result.mdi:.6f}")
if abs(mdi_manual - result.mdi) < 0.001:
    print("✓ MATCH")
else:
    print(f"✗ MISMATCH (diff: {abs(mdi_manual - result.mdi):.6f})")
print(f"{'='*80}")
