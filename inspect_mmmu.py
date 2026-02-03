#!/usr/bin/env python3
"""Quick script to inspect MMMU Pro dataset structure"""

from datasets import load_dataset

print("Loading MMMU Pro vision config...")
dataset = load_dataset("MMMU/MMMU_Pro", "vision", split="test")

print(f"\nLoaded {len(dataset)} samples")
print(f"\nFirst item keys: {list(dataset[0].keys())}")
print(f"\nFirst item content:")
for key, value in dataset[0].items():
    if key == 'image':
        print(f"  {key}: {type(value)} - {value}")
    else:
        print(f"  {key}: {value}")

print(f"\nSecond item:")
for key, value in dataset[1].items():
    if key == 'image':
        print(f"  {key}: {type(value)}")
    else:
        print(f"  {key}: {value}")
