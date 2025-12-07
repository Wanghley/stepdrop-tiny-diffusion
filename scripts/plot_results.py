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
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot points
    ax.scatter(df['throughput'], df['fid'], color=DUKE_BLUE, s=150, zorder=3, edgecolors='white', linewidth=1.5)
    
    # Add labels
    for i, row in df.iterrows():
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
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Calculate speedup relative to Baseline
    baseline_throughput = df[df['strategy'].str.contains("DDPM", case=False)]['throughput'].values[0]
    df['speedup'] = df['throughput'] / baseline_throughput
    
    bars = ax.bar(df['strategy'], df['speedup'], color=DUKE_BLUE, width=0.6, zorder=3)
    
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
        metrics['strategy'] = name
        rows.append(metrics)
    
    df = pd.DataFrame(rows)
    
    # Sort by throughput for cleaner bar charts
    # df = df.sort_values('throughput')
    
    print("Generating plots...")
    plot_pareto(df, output_dir)
    plot_speedup(df, output_dir)
    plot_flops(df, output_dir)
    
    print(f"\n🎉 Plots generated in: {output_dir}")

if __name__ == "__main__":
    main()
