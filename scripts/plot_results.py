import argparse
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Duke Colors
DUKE_BLUE = "#003087"
DUKE_ROYAL_BLUE = "#00539B"
DUKE_BLACK = "#012169" # Darker blue/black
WHITE = "#FFFFFF"
GRAY_LIGHT = "#E2E6ED"

def set_style():
    """Configures matplotlib to use a clean, academic style with Duke colors."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
    plt.rcParams['axes.edgecolor'] = DUKE_BLUE
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['text.color'] = DUKE_BLACK
    plt.rcParams['axes.labelcolor'] = DUKE_BLACK
    plt.rcParams['xtick.color'] = DUKE_BLACK
    plt.rcParams['ytick.color'] = DUKE_BLACK
    plt.rcParams['figure.dpi'] = 300

def load_latest_report(results_dir="results"):
    """Finds and loads the most recent report.json."""
    root = Path(results_dir)
    reports = sorted(root.glob("**/report.json"), key=os.path.getmtime, reverse=True)
    if not reports:
        raise FileNotFoundError("No report.json found in results/ directory!")
    
    latest_report = reports[0]
    print(f"📊 Loading report from: {latest_report}")
    with open(latest_report, "r") as f:
        data = json.load(f)
    return data, latest_report.parent

def plot_pareto(df, output_dir):
    """Generates Quality (FID) vs Speed (Throughput) scatter plot."""
    valid_df = df[(df['throughput'] > 0) & (df['fid'] > 0)].copy()
    if len(valid_df) == 0:
        print("⚠️ Skipping Pareto plot (no valid throughput/FID data)")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot points
    ax.scatter(valid_df['throughput'], valid_df['fid'], color=DUKE_BLUE, s=150, zorder=3, edgecolors='white', linewidth=1.5)
    
    # Add labels
    for i, row in valid_df.iterrows():
        ax.annotate(
            row['strategy'], 
            (row['throughput'], row['fid']), 
            xytext=(5, 5), textcoords='offset points',
            fontsize=10, fontweight='bold', color=DUKE_BLACK
        )

    ax.set_title('Speed vs. Quality (Pareto Frontier)', fontsize=14, fontweight='bold', color=DUKE_BLUE)
    ax.set_xlabel('Throughput (Images/Sec) [Higher is Better]', fontsize=11, fontweight='bold')
    ax.set_ylabel('FID Score [Lower is Better]', fontsize=11, fontweight='bold')
    
    # Invert Y axis because lower FID is better
    # ax.invert_yaxis() # Usually better not to invert for FID unless explicitly stated "Quality"
    # But Pareto usually puts "Best" in Top-Right. Let's keep it standard: Lower Y is better.
    
    # Grid
    ax.grid(True, linestyle='--', alpha=0.5, color=GRAY_LIGHT)
    
    plt.tight_layout()
    save_path = output_dir / "plot_pareto_quality_speed.png"
    plt.savefig(save_path)
    print(f"✅ Saved Pareto plot to {save_path}")
    plt.close()

def plot_speedup(df, output_dir):
    """Generates normalized speedup bar chart."""
    valid_df = df[df['throughput'] > 0].copy()
    if len(valid_df) == 0:
        print("⚠️ Skipping speedup plot (no throughput data)")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Find baseline (DDPM or first strategy)
    baseline_row = valid_df[valid_df['strategy'].str.contains("DDPM", case=False)]
    if len(baseline_row) == 0:
        baseline_row = valid_df.iloc[[0]]
    
    baseline_throughput = baseline_row['throughput'].values[0]
    valid_df['speedup'] = valid_df['throughput'] / baseline_throughput
    
    bars = ax.bar(valid_df['strategy'], valid_df['speedup'], color=DUKE_BLUE, width=0.6, zorder=3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}x',
                ha='center', va='bottom', fontweight='bold', color=DUKE_BLUE)

    ax.set_title('Generation Speedup (Relative to DDPM)', fontsize=14, fontweight='bold', color=DUKE_BLUE)
    ax.set_ylabel('Speedup Factor', fontsize=11, fontweight='bold')
    ax.set_xlabel('Strategy', fontsize=11, fontweight='bold')
    
    # Clean X labels
    plt.xticks(rotation=45, ha='right')
    
    ax.grid(axis='y', linestyle='--', alpha=0.5, color=GRAY_LIGHT)
    
    plt.tight_layout()
    save_path = output_dir / "plot_speedup.png"
    plt.savefig(save_path)
    print(f"✅ Saved Speedup plot to {save_path}")
    plt.close()

def plot_flops(df, output_dir):
    """Generates FLOPs comparison bar chart."""
    if 'flops_per_sample' not in df.columns:
        print("⚠️ Skipping FLOPs plot (no flops_per_sample data)")
        return
        
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Convert FLOPs to GFLOPs
    df['gflops'] = df['flops_per_sample'] / 1e9
    
    bars = ax.bar(df['strategy'], df['gflops'], color=DUKE_ROYAL_BLUE, width=0.6, zorder=3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}G',
                ha='center', va='bottom', fontweight='bold', color=DUKE_BLACK)

    ax.set_title('Computational Cost (FLOPs per Image)', fontsize=14, fontweight='bold', color=DUKE_BLUE)
    ax.set_ylabel('GFLOPs [Lower is Better]', fontsize=11, fontweight='bold')
    ax.set_xlabel('Strategy', fontsize=11, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.5, color=GRAY_LIGHT)
    
    plt.tight_layout()
    save_path = output_dir / "plot_flops.png"
    plt.savefig(save_path)
    print(f"✅ Saved FLOPs plot to {save_path}")
    plt.close()


def plot_metrics_comparison(df, output_dir):
    """Generates side-by-side comparison of FID and IS."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # FID plot (lower is better)
    ax1 = axes[0]
    valid_fid = df[df['fid'] > 0].copy()
    if len(valid_fid) > 0:
        bars1 = ax1.bar(valid_fid['strategy'], valid_fid['fid'], color=DUKE_BLUE, width=0.6, zorder=3)
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=9, color=DUKE_BLACK)
    ax1.set_title('FID Score [Lower is Better]', fontsize=12, fontweight='bold', color=DUKE_BLUE)
    ax1.set_ylabel('FID', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Strategy', fontsize=11, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', linestyle='--', alpha=0.5, color=GRAY_LIGHT)
    
    # IS plot (higher is better)
    ax2 = axes[1]
    valid_is = df[df['is_mean'] > 0].copy()
    if len(valid_is) > 0:
        bars2 = ax2.bar(valid_is['strategy'], valid_is['is_mean'], color=DUKE_ROYAL_BLUE, width=0.6, zorder=3, 
                       yerr=valid_is['is_std'] if 'is_std' in valid_is.columns else None, capsize=4)
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=9, color=DUKE_BLACK)
    ax2.set_title('Inception Score [Higher is Better]', fontsize=12, fontweight='bold', color=DUKE_BLUE)
    ax2.set_ylabel('IS', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Strategy', fontsize=11, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', linestyle='--', alpha=0.5, color=GRAY_LIGHT)
    
    plt.tight_layout()
    save_path = output_dir / "plot_metrics_comparison.png"
    plt.savefig(save_path)
    print(f"✅ Saved Metrics comparison plot to {save_path}")
    plt.close()


def plot_nfe_vs_quality(df, output_dir):
    """Generates NFE vs FID plot to show quality-efficiency tradeoff."""
    valid_df = df[(df['nfe'] > 0) & (df['fid'] > 0)].copy()
    if len(valid_df) == 0:
        print("⚠️ Skipping NFE vs Quality plot (no valid data)")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(valid_df['nfe'], valid_df['fid'], color=DUKE_BLUE, s=150, zorder=3, edgecolors='white', linewidth=1.5)
    
    for i, row in valid_df.iterrows():
        ax.annotate(
            row['strategy'], 
            (row['nfe'], row['fid']), 
            xytext=(5, 5), textcoords='offset points',
            fontsize=10, fontweight='bold', color=DUKE_BLACK
        )

    ax.set_title('Number of Function Evaluations vs Quality', fontsize=14, fontweight='bold', color=DUKE_BLUE)
    ax.set_xlabel('NFE (Function Evaluations) [Lower is Better]', fontsize=11, fontweight='bold')
    ax.set_ylabel('FID Score [Lower is Better]', fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5, color=GRAY_LIGHT)
    
    plt.tight_layout()
    save_path = output_dir / "plot_nfe_vs_quality.png"
    plt.savefig(save_path)
    print(f"✅ Saved NFE vs Quality plot to {save_path}")
    plt.close()


def plot_summary_table(df, output_dir):
    """Generate a summary table as an image."""
    # Select columns to show
    cols_to_show = ['strategy', 'fid', 'is_mean', 'throughput', 'nfe', 'duration']
    existing_cols = [c for c in cols_to_show if c in df.columns]
    table_df = df[existing_cols].copy()
    
    # Format numbers
    for col in ['fid', 'is_mean', 'throughput', 'duration']:
        if col in table_df.columns:
            table_df[col] = table_df[col].apply(lambda x: f"{x:.2f}" if x > 0 else "N/A")
    
    if 'nfe' in table_df.columns:
        table_df['nfe'] = table_df['nfe'].apply(lambda x: str(int(x)) if x > 0 else "N/A")
    
    # Rename columns for display
    rename_map = {
        'strategy': 'Strategy',
        'fid': 'FID ↓',
        'is_mean': 'IS ↑',
        'throughput': 'Throughput (img/s) ↑',
        'nfe': 'NFE ↓',
        'duration': 'Duration (s) ↓'
    }
    table_df = table_df.rename(columns=rename_map)
    
    fig, ax = plt.subplots(figsize=(12, max(3, len(table_df) * 0.5 + 1)))
    ax.axis('tight')
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
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(color='white', fontweight='bold')
        cell.set_edgecolor(GRAY_LIGHT)
    
    plt.title('Benchmark Results Summary', fontsize=14, fontweight='bold', color=DUKE_BLUE, pad=20)
    
    plt.tight_layout()
    save_path = output_dir / "plot_summary_table.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"✅ Saved Summary table to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate Plots for StepDrop Benchmark")
    parser.add_argument("--results", type=str, default="results", help="Root results directory")
    args = parser.parse_args()

    set_style()
    
    try:
        data, output_dir = load_latest_report(args.results)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

    # Convert dict to DataFrame
    rows = []
    for name, metrics in data.items():
        if isinstance(metrics, dict):
            # Handle nested 'name' field
            if 'name' not in metrics:
                metrics['strategy'] = name
            else:
                metrics['strategy'] = metrics.get('name', name)
            rows.append(metrics)
    
    if not rows:
        print("❌ No valid data found in report")
        return
    
    df = pd.DataFrame(rows)
    
    # Ensure required columns exist with defaults
    for col in ['fid', 'is_mean', 'is_std', 'throughput', 'nfe', 'duration']:
        if col not in df.columns:
            df[col] = -1
    
    print(f"\n📊 Loaded {len(df)} strategies")
    print(df[['strategy', 'fid', 'is_mean', 'throughput']].to_string(index=False))
    
    print("\n🎨 Generating plots...")
    
    # Generate all plots
    plot_pareto(df, output_dir)
    plot_speedup(df, output_dir)
    plot_flops(df, output_dir)
    plot_metrics_comparison(df, output_dir)
    plot_nfe_vs_quality(df, output_dir)
    plot_summary_table(df, output_dir)
    
    print(f"\n🎉 All plots generated in: {output_dir}")

if __name__ == "__main__":
    main()
