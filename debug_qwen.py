#!/usr/bin/env python3
"""Debug Qwen2.5-VL NaN issue"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from PIL import Image
import numpy as np

# Create simple test image
img = Image.new('RGB', (224, 224), color=(128, 128, 128))

from src.models.qwen_vl import Qwen2VLWrapper
from src.models.base import ModelConfig

print("="*80)
print("Qwen2.5-VL NaN Debug Test")
print("="*80)

config = ModelConfig(
    model_name="qwen2-vl",
    model_path="Qwen/Qwen2.5-VL-7B-Instruct",
    device="cuda",
    load_in_4bit=False,  # No quantization
)

wrapper = Qwen2VLWrapper(config)
print("\nLoading model...")
wrapper.load_model()
print("✓ Model loaded")

print("\nRunning forward pass...")
result = wrapper.forward_with_attention(
    text="What is in this image?",
    image=img
)

print(f"\nInput IDs shape: {result['input_ids'].shape}")
print(f"Number of attention layers: {len(result['attentions'])}")

if result['attentions']:
    attn = result['attentions'][0]
    print(f"\nFirst attention layer:")
    print(f"  Shape: {attn.shape}")
    print(f"  Device: {attn.device}")
    print(f"  Dtype: {attn.dtype}")
    print(f"  Has NaN: {torch.isnan(attn).any().item()}")
    print(f"  Has Inf: {torch.isinf(attn).any().item()}")
    print(f"  Min: {attn.min().item()}")
    print(f"  Max: {attn.max().item()}")
    print(f"  Mean: {attn.mean().item()}")

    # Check if attention is properly normalized
    row_sums = attn.sum(dim=-1)
    print(f"\nRow sums (should be ~1.0):")
    print(f"  Min: {row_sums.min().item()}")
    print(f"  Max: {row_sums.max().item()}")
    print(f"  Mean: {row_sums.mean().item()}")

    if torch.isnan(attn).any():
        print("\n⚠️ FOUND NaN in attention!")
        nan_positions = torch.where(torch.isnan(attn))
        print(f"  Number of NaN values: {len(nan_positions[0])}")
        print(f"  First NaN position: ({nan_positions[0][0].item()}, {nan_positions[1][0].item()})")
    else:
        print("\n✓ No NaN in attention")
else:
    print("\n❌ No attention weights returned!")

print("\n" + "="*80)
