#!/usr/bin/env python3
"""
Generate comprehensive plots from benchmark results.

Usage:
    python scripts/plot_results.py --results results/
    python scripts/plot_results.py --report results/2024-01-15_10-30-45/report.json
"""

import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
import seaborn as sns
from scipy import stats

# Duke Colors
DUKE_BLUE = '#012169'
DUKE_ROYAL_BLUE = '#00539B'
DUKE_NAVY = '#001A57'
DUKE_BLACK = '#262626'
GRAY_LIGHT = '#B5B5B5'
DUKE_WHITE = '#FFFFFF'

# Color palette for strategies
STRATEGY_COLORS = {
    'DDPM': '#012169',      # Duke Blue
    'DDIM': '#00539B',      # Royal Blue
    'StepDrop': '#C84E00',  # Duke Orange
    'Adaptive': '#339898',  # Teal
}

def get_strategy_color(name):
    """Get color based on strategy type."""
    for key, color in STRATEGY_COLORS.items():
        if key in name:
            return color
    return DUKE_ROYAL_BLUE


def set_style():
    """Set consistent plot style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.titlesize': 14,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.figsize': (10, 6),
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def load_latest_report(results_dir):
    """Load the most recent report.json from results_dir."""
    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
        
    # Find latest timestamped directory
    dirs = sorted([d for d in results_path.iterdir() if d.is_dir()], reverse=True)
    
    for d in dirs:
        report_path = d / "report.json"
        if report_path.exists():
            with open(report_path) as f:
                data = json.load(f)
            print(f"📂 Loaded: {report_path}")
            return data, d
    
    raise FileNotFoundError(f"No report.json found in {results_dir}")


def load_report_from_path(report_dir):
    """Load report.json from a specific directory."""
    report_path = Path(report_dir) / "report.json"
    if not report_path.exists():
         raise FileNotFoundError(f"No report.json found in {report_dir}")
    
    with open(report_path) as f:
        data = json.load(f)
    print(f"📂 Loaded: {report_path}")
    return data, Path(report_dir)


def plot_pareto(df, output_dir):
    """Plot FID vs Throughput Pareto frontier."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Filter valid data
    valid = df[(df['fid'] > 0) & (df['throughput'] > 0)].copy()
    
    if len(valid) == 0:
        print("⚠️ No valid data for Pareto plot")
        return
    
    # Plot each strategy
    for _, row in valid.iterrows():
        color = get_strategy_color(row['strategy'])
        ax.scatter(row['throughput'], row['fid'], 
                   s=150, c=color, alpha=0.8, edgecolors='white', linewidth=2,
                   zorder=3)
        ax.annotate(row['strategy'], 
                    (row['throughput'], row['fid']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, fontweight='bold')
    
    # Find and highlight Pareto frontier
    pareto_points = []
    sorted_df = valid.sort_values('throughput', ascending=False)
    min_fid = float('inf')
    for _, row in sorted_df.iterrows():
        if row['fid'] < min_fid:
            pareto_points.append((row['throughput'], row['fid']))
            min_fid = row['fid']
    
    if pareto_points:
        pareto_x, pareto_y = zip(*pareto_points)
        ax.plot(pareto_x, pareto_y, 'g--', linewidth=2, alpha=0.7, 
                label='Pareto Frontier', zorder=2)
    
    ax.set_title('Quality-Speed Pareto Analysis', fontsize=14, fontweight='bold', color=DUKE_BLUE)
    ax.set_xlabel('Throughput (images/sec) [Higher is Better ↑]', fontsize=11, fontweight='bold')
    ax.set_ylabel('FID Score [Lower is Better ↓]', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Add quadrant labels
    ax.axhline(y=df['fid'].median(), color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=df['throughput'].median(), color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    save_path = output_dir / "plot_pareto.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved Pareto plot to {save_path}")
    plt.close()


def plot_metrics_radar(df, output_dir):
    """Radar chart comparing multiple metrics."""
    # Metrics to include (normalized)
    metrics = ['fid', 'is_mean', 'throughput', 'precision', 'recall', 'vendi_score']
    available_metrics = [m for m in metrics if m in df.columns and df[m].max() > 0]
    
    if len(available_metrics) < 3:
        print("⚠️ Not enough metrics for radar plot")
        return
    
    # Normalize metrics (0-1 scale, inverted for FID)
    normalized = df.copy()
    for m in available_metrics:
        col = df[m]
        if col.max() > col.min():
            if m == 'fid':  # Lower is better
                normalized[m] = 1 - (col - col.min()) / (col.max() - col.min())
            else:  # Higher is better
                normalized[m] = (col - col.min()) / (col.max() - col.min())
        else:
            normalized[m] = 0.5
    
    # Create radar chart
    num_vars = len(available_metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    for _, row in normalized.iterrows():
        values = [row[m] for m in available_metrics]
        values += values[:1]  # Complete the loop
        
        color = get_strategy_color(row['strategy'])
        ax.plot(angles, values, 'o-', linewidth=2, label=row['strategy'], color=color)
        ax.fill(angles, values, alpha=0.1, color=color)
    
    # Format
    metric_labels = {
        'fid': 'FID (↓)',
        'is_mean': 'IS (↑)',
        'throughput': 'Speed (↑)',
        'precision': 'Precision (↑)',
        'recall': 'Recall (↑)',
        'vendi_score': 'Diversity (↑)'
    }
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([metric_labels.get(m, m) for m in available_metrics])
    ax.set_title('Strategy Comparison Radar', fontsize=14, fontweight='bold', color=DUKE_BLUE, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    save_path = output_dir / "plot_radar.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved Radar plot to {save_path}")
    plt.close()


def plot_metrics_comparison(df, output_dir):
    """Bar chart comparing all metrics across strategies."""
    metrics_config = [
        ('fid', 'FID ↓', True),
        ('is_mean', 'IS ↑', False),
        ('precision', 'Precision ↑', False),
        ('recall', 'Recall ↑', False),
        ('throughput', 'Throughput ↑', False),
    ]
    
    # Filter to available metrics
    available = [(m, l, inv) for m, l, inv in metrics_config 
                 if m in df.columns and df[m].max() > 0]
    
    if not available:
        print("⚠️ No metrics available for comparison plot")
        return
    
    fig, axes = plt.subplots(1, len(available), figsize=(4*len(available), 5))
    if len(available) == 1:
        axes = [axes]
    
    for ax, (metric, label, invert) in zip(axes, available):
        colors = [get_strategy_color(s) for s in df['strategy']]
        bars = ax.bar(range(len(df)), df[metric], color=colors, width=0.7)
        
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['strategy'], rotation=45, ha='right', fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Add value labels
        for bar, val in zip(bars, df[metric]):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Metrics Comparison', fontsize=14, fontweight='bold', color=DUKE_BLUE)
    plt.tight_layout()
    save_path = output_dir / "plot_metrics_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved Metrics Comparison to {save_path}")
    plt.close()


def plot_nfe_vs_quality(df, output_dir):
    """NFE vs FID scatter plot."""
    valid = df[(df['nfe'] > 0) & (df['fid'] > 0)].copy()
    
    if len(valid) == 0:
        print("⚠️ No valid data for NFE vs Quality plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for _, row in valid.iterrows():
        color = get_strategy_color(row['strategy'])
        ax.scatter(row['nfe'], row['fid'], s=200, c=color, 
                   alpha=0.8, edgecolors='white', linewidth=2)
        ax.annotate(row['strategy'], (row['nfe'], row['fid']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold')
    
    ax.set_title('Compute Cost vs Quality', fontsize=14, fontweight='bold', color=DUKE_BLUE)
    ax.set_xlabel('NFE (Function Evaluations) [Lower is Better ↓]', fontsize=11, fontweight='bold')
    ax.set_ylabel('FID Score [Lower is Better ↓]', fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    save_path = output_dir / "plot_nfe_vs_quality.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved NFE vs Quality plot to {save_path}")
    plt.close()


def plot_comprehensive_summary(df, output_dir):
    """Generate a comprehensive summary table as image."""
    # Select columns
    display_cols = {
        'strategy': 'Strategy',
        'fid': 'FID ↓',
        'is_mean': 'IS ↑',
        'precision': 'Prec ↑',
        'recall': 'Rec ↑',
        'vendi_score': 'Vendi ↑',
        'throughput': 'Tput ↑',
        'nfe': 'NFE ↓',
    }
    
    available_cols = [c for c in display_cols.keys() if c in df.columns]
    table_df = df[available_cols].copy()
    
    # Format numbers
    for col in table_df.columns:
        if col == 'strategy':
            continue
        elif col == 'nfe':
            table_df[col] = table_df[col].apply(lambda x: str(int(x)) if x > 0 else '-')
        else:
            table_df[col] = table_df[col].apply(lambda x: f"{x:.2f}" if x > 0 else '-')
    
    # Rename columns
    table_df.columns = [display_cols.get(c, c) for c in table_df.columns]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 2 + len(df) * 0.4))
    ax.axis('off')
    
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc='center',
        loc='center',
        colColours=[DUKE_BLUE] * len(table_df.columns)
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(color='white', fontweight='bold')
        cell.set_edgecolor(GRAY_LIGHT)
    
    plt.title('Comprehensive Benchmark Results', fontsize=14, fontweight='bold', 
              color=DUKE_BLUE, pad=20)
    
    plt.tight_layout()
    save_path = output_dir / "plot_summary_table.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved Summary Table to {save_path}")
    plt.close()


# ============================================================================
# ADVANCED PLOTTING FUNCTIONS
# ============================================================================

def plot_metric_comparison_grid(df, output_dir, figsize=(18, 12)):
    """Multi-panel comparison of key metrics."""
    metrics = [
        ('fid', 'FID Score', '↓ Lower is Better', False),
        ('is_mean', 'Inception Score', '↑ Higher is Better', True),
        ('precision', 'Precision', '↑ Higher is Better', True),
        ('recall', 'Recall', '↑ Higher is Better', True),
        ('lpips', 'LPIPS (Perceptual)', '↓ Lower is Better', False),
        ('ssim', 'SSIM', '↑ Higher is Better', True),
        ('throughput', 'Throughput (img/s)', '↑ Higher is Better', True),
        ('nfe', 'NFE (Steps)', '↓ Lower is Better', False),
    ]
    
    # Filter to available metrics
    available = [(m, l, d, h) for m, l, d, h in metrics if m in df.columns and df[m].max() > 0]
    
    if not available:
        print("⚠️ No metrics available for comparison")
        return
    
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    axes = axes.flatten()
    
    for idx, (metric, label, direction, higher_better) in enumerate(available):
        ax = axes[idx]
        
        # Prepare data
        valid_data = df[df[metric] > 0].sort_values(metric, ascending=not higher_better)
        
        if len(valid_data) == 0:
            continue
        
        colors = [get_strategy_color(s) for s in valid_data['strategy']]
        bars = ax.barh(range(len(valid_data)), valid_data[metric], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Styling
        ax.set_yticks(range(len(valid_data)))
        ax.set_yticklabels(valid_data['strategy'], fontsize=9)
        ax.set_xlabel(f'{label} {direction}', fontweight='bold', fontsize=10)
        ax.set_title(label, fontweight='bold', fontsize=11)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, valid_data[metric])):
            ax.text(val, bar.get_y() + bar.get_height()/2, 
                   f' {val:.2f}' if val < 100 else f' {val:.0f}',
                   ha='left', va='center', fontsize=8, fontweight='bold')
    
    # Hide unused subplots
    for idx in range(len(available), len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle('Comprehensive Metric Comparison Across Strategies', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    save_path = output_dir / "01_metric_comparison_grid.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {save_path}")
    plt.close()


def plot_metrics_heatmap(df, output_dir, figsize=(14, 8)):
    """Heatmap showing all metrics for each strategy."""
    # Select key metrics
    key_metrics = ['fid', 'is_mean', 'precision', 'recall', 'lpips', 'ssim', 
                   'psnr', 'vendi_score', 'intra_lpips', 'throughput', 'nfe']
    
    available_metrics = [m for m in key_metrics if m in df.columns and df[m].max() > 0]
    
    if not available_metrics:
        print("⚠️ No metrics for heatmap")
        return
    
    # Prepare data
    heatmap_data = df[['strategy'] + available_metrics].set_index('strategy')
    
    # Normalize each metric to 0-1 (inverted for "lower is better" metrics)
    normalized_heatmap = heatmap_data.copy()
    
    lower_is_better = ['fid', 'lpips', 'nfe']
    
    for col in available_metrics:
        col_data = heatmap_data[col]
        valid = col_data[col_data > 0]
        
        if len(valid) > 0:
            vmin, vmax = valid.min(), valid.max()
            if vmin < vmax:
                if col in lower_is_better:
                    normalized_heatmap[col] = 1 - (col_data - vmin) / (vmax - vmin)
                else:
                    normalized_heatmap[col] = (col_data - vmin) / (vmax - vmin)
            else:
                normalized_heatmap[col] = 0.5
        else:
            normalized_heatmap[col] = 0
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(normalized_heatmap.fillna(0).values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks(np.arange(len(available_metrics)))
    ax.set_yticks(np.arange(len(normalized_heatmap)))
    ax.set_xticklabels(available_metrics, rotation=45, ha='right', fontweight='bold')
    ax.set_yticklabels(normalized_heatmap.index, fontweight='bold')
    
    # Add values
    for i in range(len(normalized_heatmap)):
        for j in range(len(available_metrics)):
            val = heatmap_data.iloc[i, j]
            if val > 0:
                text = ax.text(j, i, f'{val:.1f}',
                             ha="center", va="center", color="black", fontsize=8, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Performance (Green=Better)', rotation=270, labelpad=20, fontweight='bold')
    
    ax.set_title('Metrics Heatmap: Strategy Performance Overview',
                fontweight='bold', fontsize=14, pad=20)
    
    plt.tight_layout()
    save_path = output_dir / "05_metrics_heatmap.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {save_path}")
    plt.close()


def plot_metric_distributions(df, output_dir, figsize=(16, 10)):
    """Box plots showing metric distributions."""
    metrics = ['fid', 'is_mean', 'lpips', 'ssim', 'psnr', 'throughput']
    available = [m for m in metrics if m in df.columns and df[m].max() > 0]
    
    if not available:
        print("⚠️ No metrics for distribution plots")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    for idx, metric in enumerate(available):
        ax = axes[idx]
        
        # Prepare data
        data_to_plot = []
        labels = []
        colors = []
        
        for _, row in df.iterrows():
            if row[metric] > 0:
                data_to_plot.append([row[metric]])
                labels.append(row['strategy'])
                colors.append(get_strategy_color(row['strategy']))
        
        if not data_to_plot:
            continue
        
        # Create violin plot
        parts = ax.violinplot(data_to_plot, positions=range(len(data_to_plot)), 
                             showmeans=True, showmedians=True)
        
        # Color the violins
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel(metric.upper(), fontweight='bold', fontsize=10)
        ax.set_title(f'Distribution of {metric.upper()}', fontweight='bold', fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Hide unused plots
    for idx in range(len(available), len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle('Metric Distribution Across Strategies', 
                fontweight='bold', fontsize=14, y=0.995)
    plt.tight_layout()
    
    save_path = output_dir / "06_metric_distributions.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {save_path}")
    plt.close()


def plot_summary_statistics(df, output_dir):
    """Generate summary statistics and print them."""
    print("\n" + "="*70)
    print("📈 BENCHMARK SUMMARY STATISTICS")
    print("="*70)
    
    if 'fid' in df.columns:
        valid_fid = df[df['fid'] > 0]['fid']
        if len(valid_fid) > 0:
            print(f"\n📊 FID Score (lower is better):")
            print(f"  🏆 Best:  {valid_fid.min():.2f} ({df.loc[valid_fid.idxmin(), 'strategy']})")
            print(f"  📉 Worst: {valid_fid.max():.2f} ({df.loc[valid_fid.idxmax(), 'strategy']})")
            print(f"  📈 Mean:  {valid_fid.mean():.2f} ± {valid_fid.std():.2f}")
    
    if 'throughput' in df.columns:
        valid_tp = df[df['throughput'] > 0]['throughput']
        if len(valid_tp) > 0:
            print(f"\n⚡ Throughput (higher is better):")
            print(f"  🏆 Best:  {valid_tp.max():.2f} img/s ({df.loc[valid_tp.idxmax(), 'strategy']})")
            print(f"  📉 Worst: {valid_tp.min():.2f} img/s ({df.loc[valid_tp.idxmin(), 'strategy']})")
            print(f"  📈 Mean:  {valid_tp.mean():.2f} ± {valid_tp.std():.2f}")
    
    if 'nfe' in df.columns:
        valid_nfe = df[df['nfe'] > 0]['nfe']
        if len(valid_nfe) > 0:
            print(f"\n⏱️  NFE - Function Evaluations (lower is faster):")
            print(f"  🏆 Min:   {valid_nfe.min():.0f}")
            print(f"  📈 Max:   {valid_nfe.max():.0f}")
            print(f"  📊 Mean:  {valid_nfe.mean():.0f} ± {valid_nfe.std():.0f}")
    
    if 'precision' in df.columns and 'recall' in df.columns:
        valid_prec = df[df['precision'] > 0]['precision']
        valid_rec = df[df['recall'] > 0]['recall']
        if len(valid_prec) > 0 and len(valid_rec) > 0:
            print(f"\n🎯 Quality Metrics:")
            print(f"  Precision: {valid_prec.mean():.4f} ± {valid_prec.std():.4f}")
            print(f"  Recall:    {valid_rec.mean():.4f} ± {valid_rec.std():.4f}")
    
    print("="*70 + "\n")


def generate_plots(report_path):
    """Generate all plots for a given report path."""
    set_style()
    
    try:
        data, output_dir = load_report_from_path(report_path)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    # Extract strategies data - handle different JSON structures
    if 'strategies' in data:
        strategies_data = data['strategies']
    else:
        # Old format - data itself is the strategies dict
        strategies_data = {k: v for k, v in data.items() 
                          if isinstance(v, dict) and 'fid' in v}
    
    if not strategies_data:
        print("❌ No valid strategy data found in report")
        return
    
    # Build DataFrame
    rows = []
    for name, metrics in strategies_data.items():
        if isinstance(metrics, dict):
            row = metrics.copy()
            row['strategy'] = row.get('name', name)
            rows.append(row)
    
    if not rows:
        print("❌ No valid strategy data found")
        return
    
    df = pd.DataFrame(rows)
    
    # Ensure numeric columns
    numeric_cols = ['fid', 'kid', 'is_mean', 'is_std', 'precision', 'recall',
                    'density', 'coverage', 'lpips', 'ssim', 'psnr',
                    'vendi_score', 'intra_lpips', 'throughput', 'nfe', 'flops']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1)
    
    print(f"\n📊 Loaded {len(df)} strategies")
    
    # Print available columns
    available_print_cols = [c for c in ['strategy', 'fid', 'is_mean', 'throughput', 'nfe'] if c in df.columns]
    print(df[available_print_cols].to_string(index=False))
    
    print("\n🎨 Generating plots...")
    
    # Generate original plots
    plot_pareto(df, output_dir)
    plot_metrics_comparison(df, output_dir)
    plot_nfe_vs_quality(df, output_dir)
    plot_metrics_radar(df, output_dir)
    plot_comprehensive_summary(df, output_dir)
    
    # Generate advanced plots
    plot_metric_comparison_grid(df, output_dir)
    plot_metrics_heatmap(df, output_dir)
    plot_metric_distributions(df, output_dir)
    
    # Print summary statistics
    plot_summary_statistics(df, output_dir)
    
    print(f"🎉 All plots saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate Benchmark Plots")
    parser.add_argument("--results", type=str, default="results",
                        help="Results directory")
    parser.add_argument("--report", type=str, default=None,
                        help="Specific report directory")
    args = parser.parse_args()
    
    set_style()
    
    if args.report:
        # Use specific report directory
        generate_plots(args.report)
    else:
        # Find latest report
        try:
            data, output_dir = load_latest_report(args.results)
            generate_plots(output_dir)
        except FileNotFoundError as e:
            print(f"❌ {e}")
            return

if __name__ == "__main__":
    main()

