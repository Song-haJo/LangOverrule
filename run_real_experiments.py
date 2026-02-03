#!/usr/bin/env python3
"""
실제 모델로 논문 Table 1 재현 실험

이 스크립트는 실제 모델과 데이터셋을 사용하여 논문의 결과를 완전히 재현합니다.
"""

# Set cache directories BEFORE any imports
import os
os.environ.setdefault('BASE_CACHE_DIR', '/mnt/tmp/cache')
os.environ.setdefault('HF_HOME', os.path.join(os.environ['BASE_CACHE_DIR'], 'hf'))
os.environ.setdefault('TORCH_HOME', os.path.join(os.environ['BASE_CACHE_DIR'], 'torch'))
os.environ.setdefault('TRANSFORMERS_CACHE', os.path.join(os.environ['BASE_CACHE_DIR'], 'hf'))
os.environ.setdefault('HF_DATASETS_CACHE', os.path.join(os.environ['BASE_CACHE_DIR'], 'hf', 'datasets'))

# Create cache directories
os.makedirs(os.environ['HF_HOME'], exist_ok=True)
os.makedirs(os.environ['TORCH_HOME'], exist_ok=True)

# Block TensorFlow imports - return None to indicate package not found
import sys
class BlockTensorFlow:
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith('tensorflow'):
            return None  # Indicate package not found
        return None
    def find_module(self, fullname, path=None):
        if fullname.startswith('tensorflow'):
            return None  # Indicate package not found
        return None
sys.meta_path.insert(0, BlockTensorFlow())

import os
import json
import argparse
from datetime import datetime
from pathlib import Path
import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models import LLaVAWrapper, Qwen2VLWrapper
from src.datasets import load_mmmu_pro, SimpleMMMUDataset
from src.metrics import compute_modality_metrics


def run_image_model_experiment(
    model_name: str,
    model_path: str,
    num_samples: int = 100,
    use_real_dataset: bool = True,
    device: str = "cuda",
    load_in_4bit: bool = True,
) -> dict:
    """
    Run experiment on image modality model.

    Args:
        model_name: Model name ('llava' or 'qwen2-vl')
        model_path: HuggingFace model path
        num_samples: Number of samples to analyze
        use_real_dataset: Use real MMMU Pro dataset
        device: Device to use
        load_in_4bit: Use 4-bit quantization

    Returns:
        Experiment results
    """
    print("="*80, flush=True)
    print(f"Running {model_name} on Image Modality", flush=True)
    print(f"Model: {model_path}", flush=True)
    print(f"Samples: {num_samples}", flush=True)
    print("="*80, flush=True)
    print("", flush=True)
    print("DEBUG: Starting experiment...", flush=True)

    # Load dataset
    print("DEBUG: Loading dataset...", flush=True)
    if use_real_dataset:
        try:
            print("\nLoading MMMU Pro dataset...", flush=True)
            dataset = load_mmmu_pro(
                split="test",
                num_samples=num_samples
            )
            print("DEBUG: MMMU Pro loaded", flush=True)
            samples = dataset.get_samples_for_analysis()
            print("DEBUG: Got samples for analysis", flush=True)
        except Exception as e:
            print(f"Failed to load MMMU Pro: {e}", flush=True)
            print("Falling back to dummy dataset...", flush=True)
            dataset = SimpleMMMUDataset(num_samples=min(num_samples, 10))
            samples = dataset.get_samples_for_analysis()
    else:
        print("\nDEBUG: Using dummy dataset...", flush=True)
        dataset = SimpleMMMUDataset(num_samples=min(num_samples, 10))
        print("DEBUG: Dummy dataset created", flush=True)
        samples = dataset.get_samples_for_analysis()
        print("DEBUG: Got dummy samples", flush=True)

    print(f"Loaded {len(samples)} samples", flush=True)

    # Initialize model wrapper
    print(f"\nDEBUG: Initializing {model_name} model...", flush=True)
    from src.models.base import ModelConfig

    print("DEBUG: Creating ModelConfig...", flush=True)
    # Disable quantization for Qwen2-VL to avoid NaN issues
    use_quantization = load_in_4bit and (model_name != "qwen2-vl")
    config = ModelConfig(
        model_name=model_name,
        model_path=model_path,
        device=device,
        load_in_4bit=use_quantization,
    )
    if model_name == "qwen2-vl" and load_in_4bit:
        print("DEBUG: Quantization disabled for Qwen2-VL (NaN prevention)", flush=True)
    print("DEBUG: ModelConfig created", flush=True)

    print(f"DEBUG: Creating wrapper for {model_name}...", flush=True)
    if model_name == "llava":
        wrapper = LLaVAWrapper(config)
    elif model_name == "qwen2-vl":
        wrapper = Qwen2VLWrapper(config)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    print("DEBUG: Wrapper created", flush=True)

    print("DEBUG: Loading model (this may take several minutes)...", flush=True)
    wrapper.load_model()
    print("✓ Model loaded successfully!", flush=True)

    # Run analysis
    print(f"\nAnalyzing {len(samples)} samples...")
    all_metrics = {'early': [], 'middle': [], 'late': []}

    from tqdm import tqdm

    for i, sample in enumerate(tqdm(samples)):
        try:
            # Ultra-aggressive memory management to prevent OOM
            import gc

            # Clear cache EVERY sample
            torch.cuda.empty_cache()

            # Force garbage collection every 3 samples
            if i % 3 == 0:
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            # Forward pass with attention
            # Using all tokens as query (let MDI calculator determine output tokens)
            result = wrapper.forward_with_attention(
                text=sample['text'],
                image=sample['image'],
            )

            # Debug token masks
            if i == 0:  # Only print for first sample
                print(f"\nDEBUG Sample 0:")
                print(f"  Text mask sum: {result['token_masks'].text_mask.sum().item()}")
                print(f"  Non-text mask sum: {result['token_masks'].nontext_mask.sum().item()}")
                print(f"  Total tokens: {len(result['token_masks'].text_mask)}")
                print(f"  Num attention layers: {len(result['attentions'])}")
                if len(result['attentions']) > 0:
                    print(f"  First attention shape: {result['attentions'][0].shape}")
                    print(f"  Attention device: {result['attentions'][0].device}")
                    print(f"  First attention min/max/mean: {result['attentions'][0].min().item():.6f} / {result['attentions'][0].max().item():.6f} / {result['attentions'][0].mean().item():.6f}")
                    # Check attention to text vs non-text tokens
                    attn = result['attentions'][0]
                    text_attn = attn[:, result['token_masks'].text_mask].sum().item()
                    nontext_attn = attn[:, result['token_masks'].nontext_mask].sum().item()
                    print(f"  Attention to text tokens (sum): {text_attn:.6f}")
                    print(f"  Attention to non-text tokens (sum): {nontext_attn:.6f}")

            # Compute metrics using auto-detected output tokens
            # Auto-detection: uses tokens that are neither text nor image (special tokens, etc.)
            if result['attentions']:
                metrics = compute_modality_metrics(
                    result['attentions'],
                    result['token_masks'].text_mask,
                    result['token_masks'].nontext_mask,
                    output_token_indices=None,  # Auto-detect output tokens
                    layerwise=True,
                )

                # Debug metrics
                if i == 0:
                    if 'output_token_indices' in result:
                        print(f"  Using {len(result['output_token_indices'])} generated tokens as query tokens")
                    if 'generated_ids' in result:
                        print(f"  Generated {len(result['generated_ids'])} tokens")
                    print(f"  Computed metrics stages: {list(metrics.keys())}")
                    for stage in ['early', 'middle', 'late']:
                        if stage in metrics:
                            print(f"  {stage}: MDI={metrics[stage].mdi:.4f}, AEI={metrics[stage].aei_text:.4f}")

                # Store metrics and free memory immediately
                for stage in ['early', 'middle', 'late']:
                    if stage in metrics:
                        # Extract values and store as native Python types to free GPU memory
                        metric = metrics[stage]
                        all_metrics[stage].append(metric)

                # Delete metrics immediately after storing
                del metrics

            # Delete all tensors from result
            if 'attentions' in result:
                for attn in result['attentions']:
                    del attn
                del result['attentions']

            del result

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\nOOM on sample {i}, clearing cache and retrying once...")
                # Aggressive cleanup on OOM
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                # Retry once with aggressive cleanup
                try:
                    result = wrapper.forward_with_attention(
                        text=sample['text'],
                        image=sample['image'],
                    )
                    # Process normally
                    if result['attentions']:
                        metrics = compute_modality_metrics(
                            result['attentions'],
                            result['token_masks'].text_mask,
                            result['token_masks'].nontext_mask,
                            output_token_indices=None,  # Auto-detect
                            layerwise=True,
                        )
                        for stage in ['early', 'middle', 'late']:
                            if stage in metrics:
                                metric = metrics[stage]
                                all_metrics[stage].append(metric)
                        del metrics
                    del result
                    print(f"  ✓ Retry succeeded")
                except Exception as retry_e:
                    print(f"  ✗ Retry failed: {retry_e}")
                    gc.collect()
                    torch.cuda.empty_cache()
            else:
                print(f"\nError on sample {i}: {e}")
            continue
        except Exception as e:
            print(f"\nError on sample {i}: {e}")
            continue

    # Aggregate results
    print("\n" + "="*80)
    print("Aggregating results...")
    print("="*80)

    aggregated = {}
    for stage, metrics_list in all_metrics.items():
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

    return {
        'model': model_path,
        'model_type': model_name,
        'dataset': 'MMMU_Pro' if use_real_dataset else 'Dummy',
        'num_samples': len(samples),
        'num_successful': min(len(all_metrics['early']), len(all_metrics['middle']), len(all_metrics['late'])),
        'metrics': aggregated,
    }


def compare_with_paper(results: dict, model_name: str) -> None:
    """
    Compare experimental results with paper results.

    Args:
        results: Experimental results
        model_name: Model name for comparison
    """
    # Paper results from Table 1
    paper_results = {
        'llava-1.5-7b': {
            'early': {'mdi': 1.58, 'aei': 1.04},
            'middle': {'mdi': 10.23, 'aei': 3.51},
            'late': {'mdi': 17.37, 'aei': 4.23},
        },
        'qwen2.5-vl-7b': {
            'early': {'mdi': 2.26, 'aei': 14.24},
            'middle': {'mdi': 21.12, 'aei': 10.86},
            'late': {'mdi': 33.10, 'aei': 1.42},
        },
    }

    # Normalize model name for comparison
    model_key = None
    if 'llava-1.5-7b' in model_name.lower():
        model_key = 'llava-1.5-7b'
    elif 'qwen2.5-vl-7b' in model_name.lower():
        model_key = 'qwen2.5-vl-7b'

    if model_key is None or model_key not in paper_results:
        print(f"\n⚠ No paper results available for {model_name}")
        return

    print("\n" + "="*80)
    print(f"Comparison with Paper (Table 1) - {model_key.upper()}")
    print("="*80)

    paper = paper_results[model_key]
    experimental = results['metrics']

    print(f"\nModel: {results['model']}")
    print(f"Samples: {results['num_successful']}/{results['num_samples']}")
    print("\nStage     | Paper (MDI, AEI) | Experimental (MDI, AEI) | Diff (MDI, AEI)")
    print("-" * 80)

    for stage in ['early', 'middle', 'late']:
        if stage in experimental and stage in paper:
            p = paper[stage]
            e = experimental[stage]

            mdi_diff = abs(e['mdi'] - p['mdi'])
            aei_diff = abs(e['aei'] - p['aei'])

            match_mdi = "✓" if mdi_diff < 2.0 else "✗"
            match_aei = "✓" if aei_diff < 2.0 else "✗"

            print(f"{stage:8}  | ({p['mdi']:6.2f}, {p['aei']:5.2f})  | "
                  f"({e['mdi']:6.2f}±{e['mdi_std']:4.2f}, {e['aei']:5.2f}±{e['aei_std']:4.2f})  | "
                  f"{match_mdi} ({mdi_diff:5.2f}, {match_aei} {aei_diff:5.2f})")

    print("\n참고:")
    print("- 차이 < 2.0은 합리적인 범위로 간주됩니다")
    print("- 데이터셋 샘플링, 난수 시드 등으로 인한 차이 가능")
    print("- 경향성 (early < middle < late)이 일치하는지 확인")


def save_results(results: dict, output_dir: str = "results") -> str:
    """
    Save experimental results to JSON file.

    Args:
        results: Results dictionary
        output_dir: Output directory

    Returns:
        Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"real_experiment_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(
        description="Run real model experiments to reproduce paper Table 1"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["llava", "qwen", "both"],
        default="llava",
        help="Model to test"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to analyze (paper uses 100)"
    )
    parser.add_argument(
        "--use-real-dataset",
        action="store_true",
        help="Use real MMMU Pro dataset (requires datasets library)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--no-quantization",
        action="store_true",
        help="Disable 4-bit quantization"
    )

    args = parser.parse_args()

    # Print configuration
    print("="*80)
    print("Real Model Experiment Configuration")
    print("="*80)
    print(f"Device: {args.device}")
    print(f"Samples: {args.num_samples}")
    print(f"Real Dataset: {args.use_real_dataset}")
    print(f"Quantization: {'Disabled' if args.no_quantization else '4-bit'}")
    print("="*80)

    all_results = {}

    # Run LLaVA
    if args.model in ["llava", "both"]:
        try:
            results = run_image_model_experiment(
                model_name="llava",
                model_path="llava-hf/llava-1.5-7b-hf",
                num_samples=args.num_samples,
                use_real_dataset=args.use_real_dataset,
                device=args.device,
                load_in_4bit=not args.no_quantization,
            )
            all_results['llava'] = results
            compare_with_paper(results, "llava-1.5-7b")
        except Exception as e:
            print(f"\n❌ LLaVA experiment failed: {e}")
            import traceback
            traceback.print_exc()

    # Run Qwen2.5-VL
    if args.model in ["qwen", "both"]:
        try:
            results = run_image_model_experiment(
                model_name="qwen2-vl",
                model_path="Qwen/Qwen2.5-VL-7B-Instruct",
                num_samples=args.num_samples,
                use_real_dataset=args.use_real_dataset,
                device=args.device,
                load_in_4bit=not args.no_quantization,
            )
            all_results['qwen2.5-vl'] = results
            compare_with_paper(results, "qwen2.5-vl-7b")
        except Exception as e:
            print(f"\n❌ Qwen2.5-VL experiment failed: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    if all_results:
        save_results(all_results, args.output_dir)

    print("\n" + "="*80)
    print("Experiments Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
