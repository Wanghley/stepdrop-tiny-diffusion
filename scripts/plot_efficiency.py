#!/usr/bin/env python3
"""
Visualize Efficiency (FLOPs & Memory) for Top Performers vs Baselines.
"""

import json
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg') # Force non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Define Duke University colors
DUKE_BLUE = "#00539B"
DUKE_ROYAL_BLUE = "#003087"
DUKE_BLACK = "#012169"  # Darker Duke Blue
DUKE_GRAY = "#B5B5B5"

def set_style():
    """Set consistent plot style."""
    plt.style.use('seaborn-v0_8-white')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 14,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'grid.alpha': 0.3,
        'figure.dpi': 300
    })

def main():
    parser = argparse.ArgumentParser(description="Generate Efficiency Plots (FLOPs/Memory)")
    parser.add_argument("--results", type=str, default="results", help="Results directory")
    args = parser.parse_args()
    
    # Locate latest report
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"❌ Path not found: {results_path}")
        return

    # Check if direct path to report dir or root results dir
    if (results_path / "report.json").exists():
        report_dir = results_path
    else:
        # Find latest timestamped directory
        dirs = sorted([d for d in results_path.iterdir() if d.is_dir() and (d/"report.json").exists()], reverse=True)
        if not dirs:
            print(f"❌ No valid report directories found in {results_path}")
            return
        report_dir = dirs[0]
    
    print(f"📂 Loading report from: {report_dir}")
    with open(report_dir / "report.json") as f:
        data = json.load(f)
        
    strategies = data['strategies']
    
    # Define selection: 2 Baselines + Top 3 strategies
    # Mapping: {ID: Label}
    selection = {
        "DDPM_1000": "DDPM_1000",
        "DDIM_50": "DDIM_50",
        "StepDrop_Target50_Importance": "SD_Imp_50",
        "StepDrop_Target50_Uniform": "SD_Uni_50",
        "StepDrop_Target25_Importance": "SD_Imp_25"
    }
    
    # Extract data
    records = []
    for key, label in selection.items():
        if key in strategies:
            s = strategies[key]
            # Calculate Total GFLOPs = (Per-Step FLOPs * NFE) / 1e9
            total_gflops = (s.get('flops', 0) * s.get('nfe', 0)) / 1e9
            
            # Memory in MB
            memory_mb = s.get('memory_gb', 0) * 1024
            
            records.append({
                "Strategy": label,
                "Total GFLOPs": total_gflops,
                "Peak Memory (MB)": memory_mb,
                "FID": s.get('fid', 0)
            })
        else:
            print(f"⚠️ Warning: Strategy {key} not found in report.")
    
    if not records:
        print("❌ No matching strategies found.")
        return

    df = pd.DataFrame(records)
    
    print("\n📊 Selected Data:")
    print(df)
    
    # Create Dual-Axis Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(df))
    width = 0.35
    
    # FLOPs Bar (Left Axis)
    bars1 = ax1.bar(x - width/2, df["Total GFLOPs"], width, label='Compute (GFLOPs)', color=DUKE_BLUE, alpha=0.9)
    ax1.set_ylabel('Total Compute (GFLOPs) [Lower is Better]', color=DUKE_BLUE, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=DUKE_BLUE)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["Strategy"], rotation=0, ha='center')
    
    # Increase Y-Limit for Headroom (Legends)
    ax1.set_ylim(0, df["Total GFLOPs"].max() * 1.25)
    
    # Memory Bar (Right Axis)
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, df["Peak Memory (MB)"], width, label='Memory (MB)', color=DUKE_GRAY, alpha=0.9)
    ax2.set_ylabel('Peak Memory (MB) [Lower is Better]', color='gray', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='gray')
    
    # Increase Y-Limit for Headroom
    ax2.set_ylim(0, df["Peak Memory (MB)"].max() * 1.25)
    
    # Add values on top of bars
    def add_labels(bars, ax, fmt='{:.0f}'):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(fmt.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    add_labels(bars1, ax1, '{:.1f}')
    add_labels(bars2, ax2, '{:.0f}')
    
    # Title and Layout
    plt.title('StepDrop Efficiency Analysis: Compute vs Memory', fontsize=16, fontweight='bold', color=DUKE_BLACK, pad=20)
    
    # Splitting Legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    output_path = report_dir / "plot_efficiency_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved plot to {output_path}")

if __name__ == "__main__":
    main()
