#!/usr/bin/env python3
"""
Script to visualize UNet evaluation results from the evaluation report.
Creates professional visualizations of per-class and per-satellite metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path

def parse_evaluation_report(report_path):
    """Parse the evaluation report and extract metrics."""
    with open(report_path, 'r') as f:
        content = f.read()
    
    # Extract overall metrics
    overall_metrics = {}
    overall_section = re.search(r'OVERALL METRICS\s+-{70}(.*?)(?:PER-CLASS|$)', content, re.DOTALL)
    if overall_section:
        for line in overall_section.group(1).strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                try:
                    overall_metrics[key.strip()] = float(value.strip())
                except ValueError:
                    pass
    
    # Extract per-class metrics
    class_metrics = {}
    class_section = re.search(r'PER-CLASS METRICS.*?-{70}\s+Class.*?-{70}\s+(.*?)(?:\n\n|PER-SATELLITE)', 
                              content, re.DOTALL)
    if class_section:
        lines = class_section.group(1).strip().split('\n')
        for line in lines:
            parts = line.split()
            if len(parts) >= 6:
                class_name = parts[0]
                try:
                    class_metrics[class_name] = {
                        'IoU': float(parts[1]),
                        'Dice': float(parts[2]),
                        'Precision': float(parts[3]),
                        'Recall': float(parts[4])
                    }
                except (ValueError, IndexError):
                    continue
    
    # Extract per-satellite metrics
    satellite_metrics = {}
    satellite_section = re.search(r'PER-SATELLITE METRICS\s+-{70}(.*?)(?:CONFUSION|$)', content, re.DOTALL)
    if satellite_section:
        sat_blocks = satellite_section.group(1).strip().split('\n\n')
        for block in sat_blocks:
            lines = block.strip().split('\n')
            if not lines:
                continue
            
            sat_name = lines[0].replace(':', '').strip()
            metrics = {}
            for line in lines[1:]:
                if 'Mean IoU:' in line:
                    metrics['IoU'] = float(line.split(':')[1].strip())
                elif 'Mean Dice:' in line:
                    metrics['Dice'] = float(line.split(':')[1].strip())
            
            if metrics:
                satellite_metrics[sat_name] = metrics
    
    return overall_metrics, class_metrics, satellite_metrics

def plot_per_class_metrics(class_metrics, output_dir):
    """Create a bar plot for per-class metrics."""
    
    # Remove background for clearer visualization
    classes = [c for c in class_metrics.keys() if c != 'background']
    metrics = ['IoU', 'Dice', 'Precision', 'Recall']
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(classes))
    width = 0.2
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    # Create bars for each metric
    for i, metric in enumerate(metrics):
        values = [class_metrics[c][metric] for c in classes]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, values, width, label=metric, color=colors[i], alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Customize plot
    ax.set_xlabel('Class', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('UNet Per-Class Performance Metrics', fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', ' ').title() for c in classes], fontsize=11)
    ax.legend(fontsize=11, loc='lower right')
    ax.set_ylim(0.85, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = output_dir / 'evaluation_per_class_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Per-class metrics plot saved to: {output_path}")
    plt.close()

def plot_per_satellite_metrics(satellite_metrics, output_dir):
    """Create visualizations for per-satellite metrics."""
    
    satellites = list(satellite_metrics.keys())
    iou_scores = [satellite_metrics[s]['IoU'] for s in satellites]
    dice_scores = [satellite_metrics[s]['Dice'] for s in satellites]
    
    # Sort by IoU score
    sorted_indices = np.argsort(iou_scores)
    satellites = [satellites[i] for i in sorted_indices]
    iou_scores = [iou_scores[i] for i in sorted_indices]
    dice_scores = [dice_scores[i] for i in sorted_indices]
    
    # Create horizontal bar plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    y_pos = np.arange(len(satellites))
    
    # Plot 1: IoU scores
    bars1 = ax1.barh(y_pos, iou_scores, color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(satellites, fontsize=11)
    ax1.set_xlabel('Mean IoU Score', fontsize=13, fontweight='bold')
    ax1.set_title('Per-Satellite IoU Performance', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlim(0.85, 0.97)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars1, iou_scores)):
        ax1.text(score + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{score:.4f}', va='center', fontsize=9, fontweight='bold')
    
    # Color code bars by performance
    for i, bar in enumerate(bars1):
        if iou_scores[i] >= 0.95:
            bar.set_color('#2ecc71')  # Green for excellent
        elif iou_scores[i] >= 0.92:
            bar.set_color('#3498db')  # Blue for good
        else:
            bar.set_color('#e67e22')  # Orange for needs improvement
    
    # Plot 2: Dice scores
    bars2 = ax2.barh(y_pos, dice_scores, color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(satellites, fontsize=11)
    ax2.set_xlabel('Mean Dice Score', fontsize=13, fontweight='bold')
    ax2.set_title('Per-Satellite Dice Performance', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlim(0.92, 0.99)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars2, dice_scores)):
        ax2.text(score + 0.0005, bar.get_y() + bar.get_height()/2, 
                f'{score:.4f}', va='center', fontsize=9, fontweight='bold')
    
    # Color code bars by performance
    for i, bar in enumerate(bars2):
        if dice_scores[i] >= 0.975:
            bar.set_color('#2ecc71')  # Green for excellent
        elif dice_scores[i] >= 0.96:
            bar.set_color('#e74c3c')  # Red for good
        else:
            bar.set_color('#e67e22')  # Orange for needs improvement
    
    plt.suptitle('UNet Segmentation Performance Across Satellites', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = output_dir / 'evaluation_per_satellite_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Per-satellite metrics plot saved to: {output_path}")
    plt.close()
    
    # Create a summary comparison plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(satellites))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, iou_scores, width, label='IoU', color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, dice_scores, width, label='Dice', color='#e74c3c', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Satellite', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('UNet Performance Comparison: IoU vs Dice Scores per Satellite', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(satellites, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=12, loc='lower right')
    ax.set_ylim(0.85, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8, rotation=0)
    
    plt.tight_layout()
    output_path = output_dir / 'evaluation_satellite_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Satellite comparison plot saved to: {output_path}")
    plt.close()

def plot_overall_summary(overall_metrics, class_metrics, satellite_metrics, output_dir):
    """Create a summary visualization with key metrics."""
    
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Overall metrics (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    metrics_to_show = ['Pixel Accuracy', 'Mean IoU', 'Mean Dice']
    values = [overall_metrics.get(m, 0) for m in metrics_to_show]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    bars = ax1.barh(metrics_to_show, values, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_xlim(0.9, 1.0)
    ax1.set_xlabel('Score', fontsize=11, fontweight='bold')
    ax1.set_title('Overall Model Performance', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    for bar, val in zip(bars, values):
        ax1.text(val - 0.002, bar.get_y() + bar.get_height()/2, 
                f'{val:.4f}', ha='right', va='center', fontsize=10, fontweight='bold', color='white')
    
    # Class-wise IoU (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    classes = [c for c in class_metrics.keys() if c != 'background']
    class_ious = [class_metrics[c]['IoU'] for c in classes]
    class_colors = ['#9b59b6', '#e67e22']
    
    bars = ax2.bar(range(len(classes)), class_ious, color=class_colors, alpha=0.8, edgecolor='black')
    ax2.set_xticks(range(len(classes)))
    ax2.set_xticklabels([c.replace('_', '\n').title() for c in classes], fontsize=10)
    ax2.set_ylabel('IoU Score', fontsize=11, fontweight='bold')
    ax2.set_title('Class-wise IoU Performance', fontsize=12, fontweight='bold')
    ax2.set_ylim(0.85, 0.95)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, class_ious):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.005, 
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Top 5 and Bottom 5 satellites (bottom)
    ax3 = fig.add_subplot(gs[1, :])
    
    satellites = list(satellite_metrics.keys())
    sat_ious = [satellite_metrics[s]['IoU'] for s in satellites]
    sorted_indices = np.argsort(sat_ious)
    
    # Get top 5 and bottom 5
    top5_indices = sorted_indices[-5:][::-1]
    bottom5_indices = sorted_indices[:5]
    
    display_sats = [satellites[i] for i in list(bottom5_indices) + list(top5_indices)]
    display_ious = [sat_ious[i] for i in list(bottom5_indices) + list(top5_indices)]
    display_colors = ['#e74c3c']*5 + ['#2ecc71']*5
    
    y_pos = np.arange(len(display_sats))
    bars = ax3.barh(y_pos, display_ious, color=display_colors, alpha=0.8, edgecolor='black')
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(display_sats, fontsize=10)
    ax3.set_xlabel('Mean IoU Score', fontsize=11, fontweight='bold')
    ax3.set_title('Satellite Performance: Bottom 5 (Red) vs Top 5 (Green)', fontsize=12, fontweight='bold')
    ax3.set_xlim(0.85, 0.97)
    ax3.axhline(y=4.5, color='gray', linestyle='--', linewidth=2, alpha=0.5)
    ax3.grid(axis='x', alpha=0.3)
    
    for bar, val in zip(bars, display_ious):
        ax3.text(val + 0.002, bar.get_y() + bar.get_height()/2, 
                f'{val:.4f}', va='center', fontsize=9, fontweight='bold')
    
    plt.suptitle('UNet Evaluation Summary - Key Performance Indicators', 
                 fontsize=15, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = output_dir / 'evaluation_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Evaluation summary plot saved to: {output_path}")
    plt.close()

def main():
    # Set up paths
    model_dir = Path(__file__).parent
    report_path = model_dir / 'evaluation_results_unet' / 'evaluation_report.txt'
    output_dir = model_dir / 'evaluation_results_unet'
    
    print("Parsing evaluation report...")
    overall_metrics, class_metrics, satellite_metrics = parse_evaluation_report(report_path)
    
    print(f"Found {len(class_metrics)} classes")
    print(f"Found {len(satellite_metrics)} satellites")
    
    print("\nGenerating evaluation plots...")
    plot_per_class_metrics(class_metrics, output_dir)
    plot_per_satellite_metrics(satellite_metrics, output_dir)
    plot_overall_summary(overall_metrics, class_metrics, satellite_metrics, output_dir)
    
    print("\nDone! All evaluation plots generated successfully.")
    print(f"Plots saved in: {output_dir}")

if __name__ == '__main__':
    main()
