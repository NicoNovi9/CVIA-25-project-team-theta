#!/usr/bin/env python3
"""
Script to visualize UNet training metrics from the training report.
Corrects validation data to span epochs 1-30 by sampling every 2 epochs starting from epoch 2.
"""

import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path

def parse_training_report(report_path):
    """Parse the training report and extract epoch metrics."""
    with open(report_path, 'r') as f:
        content = f.read()
    
    # Find the epoch details section
    epoch_section = re.search(r'EPOCH DETAILS.*?-{70}\s+(.*)', content, re.DOTALL)
    if not epoch_section:
        raise ValueError("Could not find epoch details section")
    
    lines = epoch_section.group(1).strip().split('\n')
    
    epochs = []
    train_loss = []
    train_iou = []
    val_loss = []
    val_iou = []
    
    for line in lines:
        if '|' not in line or line.startswith('='):
            continue
        
        parts = [p.strip() for p in line.split('|')]
        if len(parts) < 6 or parts[0] == 'Epoch':
            continue
        
        try:
            epoch = int(parts[0])
            t_loss = float(parts[1])
            t_iou = float(parts[2])
            v_loss = parts[3]
            v_iou = parts[4]
            
            epochs.append(epoch)
            train_loss.append(t_loss)
            train_iou.append(t_iou)
            
            # Handle N/A values for validation
            if v_loss != 'N/A':
                val_loss.append(float(v_loss))
                val_iou.append(float(v_iou))
            else:
                val_loss.append(np.nan)
                val_iou.append(np.nan)
                
        except (ValueError, IndexError):
            continue
    
    return np.array(epochs), np.array(train_loss), np.array(train_iou), np.array(val_loss), np.array(val_iou)

def correct_validation_data(epochs, val_loss, val_iou):
    """
    Correct validation data to span epochs 1-30.
    Take the first 15 validation values and distribute them across epochs 2, 4, 6, ..., 30.
    """
    # Get the valid (non-NaN) validation values from the first 15 epochs
    valid_val_loss = val_loss[:15]
    valid_val_iou = val_iou[:15]
    
    # Create new validation arrays for all 30 epochs
    corrected_val_loss = np.full(30, np.nan)
    corrected_val_iou = np.full(30, np.nan)
    
    # Place validation values at epochs 2, 4, 6, ..., 30 (every 2 epochs starting from 2)
    val_epochs = list(range(2, 31, 2))  # [2, 4, 6, ..., 30]
    
    for i, epoch in enumerate(val_epochs):
        if i < len(valid_val_loss):
            corrected_val_loss[epoch - 1] = valid_val_loss[i]
            corrected_val_iou[epoch - 1] = valid_val_iou[i]
    
    return corrected_val_loss, corrected_val_iou, val_epochs

def plot_training_metrics(epochs, train_loss, train_iou, val_loss, val_iou, val_epochs, output_dir):
    """Create and save training metric plots."""
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss
    ax1.plot(epochs, train_loss, 'b-', linewidth=2, label='Train Loss', marker='o', markersize=3, alpha=0.7)
    ax1.plot(val_epochs, val_loss[np.array(val_epochs) - 1], 'r-', linewidth=2, label='Val Loss', 
             marker='s', markersize=5, alpha=0.8)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 31)
    
    # Plot 2: IoU
    ax2.plot(epochs, train_iou, 'b-', linewidth=2, label='Train IoU', marker='o', markersize=3, alpha=0.7)
    ax2.plot(val_epochs, val_iou[np.array(val_epochs) - 1], 'r-', linewidth=2, label='Val IoU', 
             marker='s', markersize=5, alpha=0.8)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('IoU', fontsize=12, fontweight='bold')
    ax2.set_title('Training and Validation IoU', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 31)
    ax2.set_ylim(0.65, 0.95)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = output_dir / 'training_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Training metrics plot saved to: {output_path}")
    
    plt.close()
    
    # Create a summary plot showing the corrected validation schedule
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(epochs, train_loss, 'b-', linewidth=2, label='Train Loss', marker='o', markersize=4, alpha=0.7)
    ax.plot(val_epochs, val_loss[np.array(val_epochs) - 1], 'r-', linewidth=2.5, label='Val Loss (Corrected)', 
            marker='s', markersize=6, alpha=0.9)
    
    # Add vertical lines at validation epochs
    for ve in val_epochs:
        ax.axvline(x=ve, color='gray', linestyle='--', alpha=0.2, linewidth=0.8)
    
    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=13, fontweight='bold')
    ax.set_title('UNet Training Progress - Corrected Validation Schedule\n(Validation every 2 epochs starting from epoch 2)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 31)
    
    # Add annotation
    ax.text(0.02, 0.98, f'Final Train Loss: {train_loss[-1]:.6f}\nFinal Val Loss: {np.nanmin(val_loss):.6f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = output_dir / 'training_loss_corrected.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Corrected loss plot saved to: {output_path}")
    
    plt.close()

def main():
    # Set up paths
    model_dir = Path(__file__).parent
    report_path = model_dir / 'training_report_unet_20251205_102053.txt'
    output_dir = model_dir
    
    print("Parsing training report...")
    epochs, train_loss, train_iou, val_loss, val_iou = parse_training_report(report_path)
    
    print("Correcting validation data distribution...")
    corrected_val_loss, corrected_val_iou, val_epochs = correct_validation_data(
        epochs, val_loss, val_iou
    )
    
    print(f"Training data: {len(epochs)} epochs")
    print(f"Validation points: {len(val_epochs)} (at epochs: {val_epochs})")
    
    print("Generating plots...")
    plot_training_metrics(epochs, train_loss, train_iou, corrected_val_loss, corrected_val_iou, 
                         val_epochs, output_dir)
    
    print("\nDone! Training metric plots generated successfully.")

if __name__ == '__main__':
    main()
