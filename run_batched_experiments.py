#!/usr/bin/env python3
"""
배치 실험 스크립트 - 메모리 관리를 위해 배치로 나눠서 실행
"""

# Block TensorFlow FIRST, before any imports
import sys
class BlockTensorFlow:
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith('tensorflow'):
            return None
        return None
    def find_module(self, fullname, path=None):
        if fullname.startswith('tensorflow'):
            return None
        return None
sys.meta_path.insert(0, BlockTensorFlow())

# Set cache directories and TF environment
import os
os.environ['BASE_CACHE_DIR'] = '/mnt/tmp/cache'
os.environ['HF_HOME'] = os.path.join(os.environ['BASE_CACHE_DIR'], 'hf')
os.environ['TORCH_HOME'] = os.path.join(os.environ['BASE_CACHE_DIR'], 'torch')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.environ['BASE_CACHE_DIR'], 'hf')
os.environ['HF_DATASETS_CACHE'] = os.path.join(os.environ['BASE_CACHE_DIR'], 'hf', 'datasets')
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['USE_TF'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Create cache directories
os.makedirs(os.environ['HF_HOME'], exist_ok=True)
os.makedirs(os.environ['TORCH_HOME'], exist_ok=True)

import json
import argparse
from datetime import datetime
from pathlib import Path
import torch
import numpy as np
import gc

sys.path.insert(0, str(Path(__file__).parent))

from src.models import LLaVAWrapper
from src.datasets import load_mmmu_pro
from src.metrics import compute_modality_metrics
from src.models.base import ModelConfig


def run_batch(
    model_path: str,
    samples: list,
    batch_id: int,
    device: str = "cuda",
    load_in_4bit: bool = True,
):
    """Run a single batch of samples."""
    print(f"\n{'='*80}")
    print(f"Batch {batch_id}: Processing {len(samples)} samples")
    print(f"{'='*80}\n")

    # Create fresh model wrapper for this batch
    config = ModelConfig(
        model_name="llava",
        model_path=model_path,
        device=device,
        load_in_4bit=load_in_4bit,
    )
    wrapper = LLaVAWrapper(config)

    print(f"Loading model for batch {batch_id}...")
    wrapper.load_model()
    print("✓ Model loaded!\n")

    all_metrics = {'early': [], 'middle': [], 'late': []}
    success_count = 0

    from tqdm import tqdm

    for i, sample in enumerate(tqdm(samples, desc=f"Batch {batch_id}")):
        try:
            # Clear cache periodically
            if i % 10 == 0:
                torch.cuda.empty_cache()

            # Forward pass
            result = wrapper.forward_with_attention(
                text=sample['text'],
                image=sample['image'],
            )

            # Compute metrics
            if result['attentions']:
                text_indices = torch.where(result['token_masks'].text_mask)[0]

                metrics = compute_modality_metrics(
                    result['attentions'],
                    result['token_masks'].text_mask,
                    result['token_masks'].nontext_mask,
                    output_token_indices=text_indices,
                    layerwise=True,
                )

                for stage in ['early', 'middle', 'late']:
                    if stage in metrics:
                        all_metrics[stage].append(metrics[stage])

                success_count += 1
                del metrics
                del text_indices

            # Cleanup
            if 'attentions' in result:
                for attn in result['attentions']:
                    del attn
                del result['attentions']
            del result

            if i % 20 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\nOOM on sample {i}, clearing and continuing...")
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            else:
                print(f"\nError on sample {i}: {e}")
            continue
        except Exception as e:
            print(f"\nError on sample {i}: {e}")
            continue

    # Clean up model completely
    del wrapper
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print(f"\nBatch {batch_id} complete: {success_count}/{len(samples)} successful\n")

    return all_metrics, success_count


def main():
    parser = argparse.ArgumentParser(description="Run batched experiments")
    parser.add_argument("--total-samples", type=int, default=300, help="Total samples to process")
    parser.add_argument("--batch-size", type=int, default=100, help="Samples per batch")
    parser.add_argument("--model", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--output-dir", type=str, default="results")

    args = parser.parse_args()

    print("="*80)
    print("Batched Experiment Configuration")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Total samples: {args.total_samples}")
    print(f"Batch size: {args.batch_size}")
    print("="*80)

    # Load dataset once
    print("\nLoading MMMU Pro dataset...")
    dataset = load_mmmu_pro(split="test", num_samples=args.total_samples)
    all_samples = dataset.get_samples_for_analysis()
    print(f"Loaded {len(all_samples)} samples\n")

    # Split into batches
    num_batches = (len(all_samples) + args.batch_size - 1) // args.batch_size
    batches = []
    for i in range(num_batches):
        start_idx = i * args.batch_size
        end_idx = min((i + 1) * args.batch_size, len(all_samples))
        batches.append(all_samples[start_idx:end_idx])

    print(f"Split into {len(batches)} batches\n")

    # Process each batch
    combined_metrics = {'early': [], 'middle': [], 'late': []}
    total_success = 0

    for batch_id, batch_samples in enumerate(batches):
        metrics, success = run_batch(
            model_path=args.model,
            samples=batch_samples,
            batch_id=batch_id,
            device="cuda",
            load_in_4bit=True,
        )

        # Combine metrics
        for stage in ['early', 'middle', 'late']:
            combined_metrics[stage].extend(metrics[stage])

        total_success += success

        print(f"Progress: {total_success}/{len(all_samples)} total successful samples\n")

    # Aggregate results
    print("\n" + "="*80)
    print("Final Aggregated Results")
    print("="*80)

    aggregated = {}
    for stage, metrics_list in combined_metrics.items():
        if metrics_list:
            mdi_values = [m.mdi for m in metrics_list]
            aei_values = [m.aei_text for m in metrics_list]

            aggregated[stage] = {
                'mdi': float(np.mean(mdi_values)),
                'mdi_std': float(np.std(mdi_values)),
                'aei': float(np.mean(aei_values)),
                'aei_std': float(np.std(aei_values)),
                'num_samples': len(metrics_list),
            }

            print(f"\n{stage.upper()} layers:")
            print(f"  MDI: {aggregated[stage]['mdi']:.2f} ± {aggregated[stage]['mdi_std']:.2f}")
            print(f"  AEI: {aggregated[stage]['aei']:.2f} ± {aggregated[stage]['aei_std']:.2f}")
            print(f"  Samples: {aggregated[stage]['num_samples']}")

    # Compare with paper
    paper_results = {
        'early': {'mdi': 1.58, 'aei': 1.04},
        'middle': {'mdi': 10.23, 'aei': 3.51},
        'late': {'mdi': 17.37, 'aei': 4.23},
    }

    print("\n" + "="*80)
    print("Comparison with Paper (Table 1) - LLAVA-1.5-7B")
    print("="*80)
    print(f"\nModel: {args.model}")
    print(f"Samples: {total_success}/{len(all_samples)}")
    print("\nStage     | Paper (MDI, AEI) | Experimental (MDI, AEI) | Diff (MDI, AEI)")
    print("-" * 80)

    for stage in ['early', 'middle', 'late']:
        if stage in aggregated and stage in paper_results:
            p = paper_results[stage]
            e = aggregated[stage]

            mdi_diff = abs(e['mdi'] - p['mdi'])
            aei_diff = abs(e['aei'] - p['aei'])

            match_mdi = "✓" if mdi_diff < 2.0 else "✗"
            match_aei = "✓" if aei_diff < 2.0 else "✗"

            print(f"{stage:8}  | ({p['mdi']:6.2f}, {p['aei']:5.2f})  | "
                  f"({e['mdi']:6.2f}±{e['mdi_std']:4.2f}, {e['aei']:5.2f}±{e['aei_std']:4.2f})  | "
                  f"{match_mdi} ({mdi_diff:5.2f}, {match_aei} {aei_diff:5.2f})")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"batched_experiment_{timestamp}.json"
    filepath = os.path.join(args.output_dir, filename)

    results = {
        'model': args.model,
        'total_samples': len(all_samples),
        'successful_samples': total_success,
        'metrics': aggregated,
    }

    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {filepath}")
    print("\n" + "="*80)
    print("Experiments Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
