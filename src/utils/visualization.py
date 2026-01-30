"""
Visualization utilities for text dominance analysis.

Provides functions to create figures similar to those in the paper:
- Figure 2: MDI comparison across modalities
- Figure 3: MDI and AEI comparison between models
- Figure 4: MDI and AEI with token scaling
- Attention heatmaps
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataclasses import dataclass


@dataclass
class PlotConfig:
    """Configuration for plots."""
    figsize: Tuple[int, int] = (10, 6)
    dpi: int = 100
    font_size: int = 12
    title_size: int = 14
    colors: Dict[str, str] = None

    def __post_init__(self):
        if self.colors is None:
            self.colors = {
                'image': '#FF6B6B',
                'video': '#4ECDC4',
                'audio': '#45B7D1',
                'timeseries': '#96CEB4',
                'graph': '#FFEAA7',
                'text': '#DDA0DD',
                'early': '#3498db',
                'middle': '#2ecc71',
                'late': '#e74c3c',
            }


def plot_mdi_comparison(
    results: Dict[str, Dict[str, float]],
    config: Optional[PlotConfig] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot MDI comparison across different models/modalities.

    Similar to Figure 2 in the paper.

    Args:
        results: Dictionary mapping model names to their MDI values
                 e.g., {'LLaVA-1.5-7B': {'early': 1.58, 'middle': 10.23, 'late': 17.37}}
        config: Plot configuration
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure
    """
    config = config or PlotConfig()

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    models = list(results.keys())
    stages = ['early', 'middle', 'late']
    x = np.arange(len(models))
    width = 0.25

    for i, stage in enumerate(stages):
        values = [results[model].get(stage, 0) for model in models]
        bars = ax.bar(
            x + i * width,
            values,
            width,
            label=stage.capitalize(),
            color=config.colors[stage],
        )

    # Add MDI = 1 reference line
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, label='MDI = 1 (balanced)')

    ax.set_xlabel('Model', fontsize=config.font_size)
    ax.set_ylabel('Modality Dominance Index (MDI)', fontsize=config.font_size)
    ax.set_title('Text Dominance Across Models and Layers', fontsize=config.title_size)
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.set_yscale('log')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=config.dpi, bbox_inches='tight')

    return fig


def plot_attention_heatmap(
    attention_matrix: np.ndarray,
    text_range: Tuple[int, int],
    nontext_range: Tuple[int, int],
    config: Optional[PlotConfig] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot attention heatmap with modality boundaries.

    Args:
        attention_matrix: 2D attention matrix (seq_len, seq_len)
        text_range: (start, end) indices for text tokens
        nontext_range: (start, end) indices for non-text tokens
        config: Plot configuration
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure
    """
    config = config or PlotConfig()

    fig, ax = plt.subplots(figsize=(8, 8), dpi=config.dpi)

    # Plot heatmap
    im = ax.imshow(attention_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax, label='Attention Weight')

    # Add boundary lines for modalities
    for boundary in [text_range[0], text_range[1], nontext_range[0], nontext_range[1]]:
        ax.axvline(x=boundary - 0.5, color='red', linestyle='--', linewidth=0.5)
        ax.axhline(y=boundary - 0.5, color='red', linestyle='--', linewidth=0.5)

    ax.set_xlabel('Key Position', fontsize=config.font_size)
    ax.set_ylabel('Query Position', fontsize=config.font_size)
    ax.set_title('Cross-Modal Attention Distribution', fontsize=config.title_size)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=config.dpi, bbox_inches='tight')

    return fig


def plot_layer_metrics(
    layer_mdi: List[float],
    layer_aei: List[float],
    model_name: str = "Model",
    config: Optional[PlotConfig] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot MDI and AEI across layers.

    Similar to Figure 3 in the paper.

    Args:
        layer_mdi: MDI values per layer
        layer_aei: AEI values per layer
        model_name: Name of the model
        config: Plot configuration
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure
    """
    config = config or PlotConfig()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=config.dpi)

    layers = range(len(layer_mdi))

    # MDI plot
    ax1.plot(layers, layer_mdi, 'o-', color='#3498db', linewidth=2, markersize=6)
    ax1.axhline(y=1, color='gray', linestyle='--', linewidth=1, label='MDI = 1')
    ax1.set_ylabel('MDI', fontsize=config.font_size)
    ax1.set_title(f'{model_name} - Modality Dominance Index per Layer', fontsize=config.title_size)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # AEI plot
    ax2.plot(layers, layer_aei, 's-', color='#e74c3c', linewidth=2, markersize=6)
    ax2.axhline(y=1, color='gray', linestyle='--', linewidth=1, label='AEI = 1')
    ax2.set_xlabel('Layer', fontsize=config.font_size)
    ax2.set_ylabel('AEI (Text)', fontsize=config.font_size)
    ax2.set_title(f'{model_name} - Attention Efficiency Index per Layer', fontsize=config.title_size)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=config.dpi, bbox_inches='tight')

    return fig


def plot_modality_distribution(
    modality_results: Dict[str, Dict[str, float]],
    config: Optional[PlotConfig] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot MDI distribution across modalities.

    Creates a horizontal bar chart similar to Figure 2 in the paper.

    Args:
        modality_results: Dictionary mapping modality names to their late-layer MDI
                         e.g., {'Image': 17.37, 'Video': 157.53, 'Audio': 1.16}
        config: Plot configuration
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure
    """
    config = config or PlotConfig()

    fig, ax = plt.subplots(figsize=(10, 6), dpi=config.dpi)

    modalities = list(modality_results.keys())
    mdi_values = list(modality_results.values())

    # Sort by MDI value
    sorted_pairs = sorted(zip(modalities, mdi_values), key=lambda x: x[1])
    modalities, mdi_values = zip(*sorted_pairs)

    y_pos = np.arange(len(modalities))

    # Create horizontal bars with colors based on modality
    colors = [config.colors.get(m.lower(), '#95a5a6') for m in modalities]
    bars = ax.barh(y_pos, mdi_values, color=colors)

    # Add MDI = 1 reference line
    ax.axvline(x=1, color='black', linestyle='--', linewidth=2, label='MDI = 1')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(modalities)
    ax.set_xlabel('Modality Dominance Index (MDI)', fontsize=config.font_size)
    ax.set_title('Late-Layer MDI Across Modalities', fontsize=config.title_size)
    ax.set_xscale('log')
    ax.legend()

    # Add value labels on bars
    for bar, val in zip(bars, mdi_values):
        ax.text(
            bar.get_width() * 1.1, bar.get_y() + bar.get_height() / 2,
            f'{val:.2f}', va='center', fontsize=10
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=config.dpi, bbox_inches='tight')

    return fig


def plot_token_scaling_effect(
    scaling_results: Dict[int, Dict[str, float]],
    modality_name: str = "Audio",
    config: Optional[PlotConfig] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot effect of token replication on MDI and AEI.

    Similar to Figure 4 in the paper.

    Args:
        scaling_results: Dictionary mapping replication factor to metrics
                        e.g., {1: {'mdi_late': 1.16, 'aei_late': 1.08}, 5: {...}, 10: {...}}
        modality_name: Name of the modality
        config: Plot configuration
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure
    """
    config = config or PlotConfig()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=config.dpi)

    factors = sorted(scaling_results.keys())
    stages = ['early', 'middle', 'late']

    # MDI subplot
    for stage in stages:
        mdi_values = [scaling_results[f].get(f'mdi_{stage}', 0) for f in factors]
        ax1.plot(factors, mdi_values, 'o-', label=stage.capitalize(),
                color=config.colors[stage], linewidth=2, markersize=8)

    ax1.axhline(y=1, color='gray', linestyle='--', linewidth=1)
    ax1.set_xlabel('Token Replication Factor', fontsize=config.font_size)
    ax1.set_ylabel('MDI', fontsize=config.font_size)
    ax1.set_title(f'{modality_name}: MDI with Token Scaling', fontsize=config.title_size)
    ax1.legend()
    ax1.set_xticks(factors)
    ax1.grid(True, alpha=0.3)

    # AEI subplot
    for stage in stages:
        aei_values = [scaling_results[f].get(f'aei_{stage}', 0) for f in factors]
        ax2.plot(factors, aei_values, 's-', label=stage.capitalize(),
                color=config.colors[stage], linewidth=2, markersize=8)

    ax2.axhline(y=1, color='gray', linestyle='--', linewidth=1)
    ax2.set_xlabel('Token Replication Factor', fontsize=config.font_size)
    ax2.set_ylabel('AEI (Text)', fontsize=config.font_size)
    ax2.set_title(f'{modality_name}: AEI with Token Scaling', fontsize=config.title_size)
    ax2.legend()
    ax2.set_xticks(factors)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=config.dpi, bbox_inches='tight')

    return fig


def create_results_table(
    results: Dict[str, Dict[str, Dict[str, float]]],
    caption: str = "Modality Dominance Analysis Results",
) -> str:
    """
    Create a formatted table similar to Table 1 in the paper.

    Args:
        results: Nested dictionary with structure:
                 {model: {dataset: {'mdi_early': x, 'aei_early': y, ...}}}
        caption: Table caption

    Returns:
        Formatted table string
    """
    lines = []
    lines.append("=" * 100)
    lines.append(caption)
    lines.append("=" * 100)

    header = f"{'Model':<25} {'Dataset':<25} {'Early':^15} {'Middle':^15} {'Late':^15}"
    subheader = f"{'':<50} {'MDI':>7} {'AEI':>7} {'MDI':>7} {'AEI':>7} {'MDI':>7} {'AEI':>7}"
    lines.append(header)
    lines.append(subheader)
    lines.append("-" * 100)

    for model, datasets in results.items():
        for dataset, metrics in datasets.items():
            mdi_e = metrics.get('mdi_early', 0)
            aei_e = metrics.get('aei_early', 0)
            mdi_m = metrics.get('mdi_middle', 0)
            aei_m = metrics.get('aei_middle', 0)
            mdi_l = metrics.get('mdi_late', 0)
            aei_l = metrics.get('aei_late', 0)

            line = f"{model:<25} {dataset:<25} {mdi_e:>7.2f} {aei_e:>7.2f} {mdi_m:>7.2f} {aei_m:>7.2f} {mdi_l:>7.2f} {aei_l:>7.2f}"
            lines.append(line)

    lines.append("=" * 100)
    return "\n".join(lines)


def plot_compression_effect(
    compression_results: Dict[float, Dict[str, float]],
    model_name: str = "LLaVA-1.5-7B",
    config: Optional[PlotConfig] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot effect of token compression on MDI.

    Similar to Table 2 in the paper.

    Args:
        compression_results: Dict mapping reduction rate to metrics
                            e.g., {0.0: {'mdi_late': 17.37}, 0.75: {'mdi_late': 3.39}, ...}
        model_name: Name of the model
        config: Plot configuration
        save_path: Optional path to save

    Returns:
        Matplotlib figure
    """
    config = config or PlotConfig()

    fig, ax = plt.subplots(figsize=(10, 6), dpi=config.dpi)

    rates = sorted(compression_results.keys())
    stages = ['early', 'middle', 'late']

    x = np.arange(len(rates))
    width = 0.25

    for i, stage in enumerate(stages):
        mdi_values = [compression_results[r].get(f'mdi_{stage}', 0) for r in rates]
        ax.bar(x + i * width, mdi_values, width, label=stage.capitalize(),
               color=config.colors[stage])

    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, label='MDI = 1')

    ax.set_xlabel('Token Reduction Rate (%)', fontsize=config.font_size)
    ax.set_ylabel('MDI', fontsize=config.font_size)
    ax.set_title(f'{model_name}: Effect of Token Compression on Text Dominance',
                fontsize=config.title_size)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'{int(r*100)}%' for r in rates])
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=config.dpi, bbox_inches='tight')

    return fig
