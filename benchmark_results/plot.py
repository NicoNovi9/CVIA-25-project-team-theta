#!/usr/bin/env python3
"""
Strong Scaling Benchmark Visualization Script
==============================================
Generates meaningful plots for strong scaling experiments.

Strong Scaling: Fixed global batch size (32), varying GPUs
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory for plots
os.makedirs('plots', exist_ok=True)

# Load data
df = pd.read_csv('training_summary.csv')

# Strong scaling: rows 0-3 (global_batch_size = 32)
strong_scaling = df.iloc[0:4].copy()

print("Strong Scaling Data:")
print(strong_scaling[['num_gpus', 'global_batch_size', 'batch_size_per_gpu', 'samples_per_sec', 'avg_epoch_time_sec']])

# Calculate speedup for strong scaling (relative to 1 GPU)
base_time_strong = strong_scaling.iloc[0]['avg_epoch_time_sec']
strong_scaling['speedup'] = base_time_strong / strong_scaling['avg_epoch_time_sec']
strong_scaling['ideal_speedup'] = strong_scaling['num_gpus']
strong_scaling['efficiency'] = (strong_scaling['speedup'] / strong_scaling['num_gpus']) * 100

# Plot styling
plt.style.use('seaborn-v0_8-whitegrid')
colors = {'strong': '#2E86AB', 'ideal': '#F18F01', 'efficiency': '#C73E1D'}

# ===========================================================================
# PLOT 1: Strong Scaling - Speedup
# ===========================================================================
fig, ax = plt.subplots(figsize=(10, 7))

gpus = strong_scaling['num_gpus'].values
speedup = strong_scaling['speedup'].values
ideal = strong_scaling['ideal_speedup'].values

ax.plot(gpus, ideal, 'o--', color=colors['ideal'], linewidth=2, markersize=10, label='Ideal Speedup')
ax.plot(gpus, speedup, 's-', color=colors['strong'], linewidth=2.5, markersize=12, label='Measured Speedup')

# Add efficiency annotations
# for i, (g, s, e) in enumerate(zip(gpus, speedup, strong_scaling['efficiency'].values)):
#     ax.annotate(f'{e:.1f}%', (g, s), textcoords="offset points", xytext=(0, 15), 
#                 ha='center', fontsize=11, color=colors['strong'], fontweight='bold')

ax.set_xlabel('Number of GPUs', fontsize=14)
ax.set_ylabel('Speedup', fontsize=14)
ax.set_title('Strong Scaling: Speedup vs Number of GPUs\n(Global Batch Size = 32)', fontsize=16, fontweight='bold')
ax.set_xticks(gpus)
ax.set_xticklabels([str(g) for g in gpus])
ax.legend(loc='upper left', fontsize=12)
ax.set_xlim(0, max(gpus) + 1)
ax.set_ylim(0, max(ideal) + 1)
ax.grid(True, alpha=0.3)

# Vertical line: transition from 1 to 2 nodes (between 4 and 8 GPUs)
ax.axvline(x=6, color='gray', linestyle='--', linewidth=2, alpha=0.8, label='1→2 nodes')
ax.legend(loc='upper left', fontsize=12)

plt.tight_layout()
plt.savefig('plots/strong_scaling_speedup.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plots/strong_scaling_speedup.png")

# ===========================================================================
# PLOT 2: Strong Scaling - Epoch Time
# ===========================================================================
fig, ax = plt.subplots(figsize=(10, 7))

gpus = strong_scaling['num_gpus'].values
epoch_time = strong_scaling['avg_epoch_time_sec'].values
ideal_time = strong_scaling.iloc[0]['avg_epoch_time_sec'] / gpus

ax.plot(gpus, ideal_time, 'o--', color=colors['ideal'], linewidth=2, markersize=10, label='Ideal Time')
ax.plot(gpus, epoch_time, 's-', color=colors['strong'], linewidth=2.5, markersize=12, label='Measured Time')

ax.set_xlabel('Number of GPUs', fontsize=14)
ax.set_ylabel('Time per Epoch (seconds)', fontsize=14)
ax.set_title('Strong Scaling: Time per Epoch vs Number of GPUs\n(Global Batch Size = 32)', fontsize=16, fontweight='bold')
ax.set_xticks(gpus)
ax.set_xticklabels([str(g) for g in gpus])
ax.legend(loc='upper right', fontsize=12)
ax.set_xlim(0, max(gpus) + 1)
ax.grid(True, alpha=0.3)

# Vertical line: transition from 1 to 2 nodes (between 4 and 8 GPUs)
ax.axvline(x=6, color='gray', linestyle='--', linewidth=2, alpha=0.8, label='1→2 nodes')
ax.legend(loc='upper right', fontsize=12)

plt.tight_layout()
plt.savefig('plots/strong_scaling_epoch_time.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plots/strong_scaling_epoch_time.png")

# ===========================================================================
# PLOT 3: Strong Scaling - Throughput
# ===========================================================================
# fig, ax = plt.subplots(figsize=(10, 7))

# gpus = strong_scaling['num_gpus'].values
# throughput = strong_scaling['samples_per_sec'].values
# ideal_throughput = strong_scaling.iloc[0]['samples_per_sec'] * gpus

# ax.plot(gpus, ideal_throughput, 'o--', color=colors['ideal'], linewidth=2, markersize=10, label='Ideal Throughput')
# ax.plot(gpus, throughput, 's-', color=colors['strong'], linewidth=2.5, markersize=12, label='Measured Throughput')

# ax.set_xlabel('Number of GPUs', fontsize=14)
# ax.set_ylabel('Throughput (samples/sec)', fontsize=14)
# ax.set_title('Strong Scaling: Throughput vs Number of GPUs\n(Global Batch Size = 32)', fontsize=16, fontweight='bold')
# ax.set_xticks(gpus)
# ax.set_xticklabels([str(g) for g in gpus])
# ax.legend(loc='upper left', fontsize=12)
# ax.set_xlim(0, max(gpus) + 1)
# ax.grid(True, alpha=0.3)

# # Vertical line: transition from 1 to 2 nodes (between 4 and 8 GPUs)
# ax.axvline(x=6, color='gray', linestyle='--', linewidth=2, alpha=0.8, label='1→2 nodes')
# ax.legend(loc='upper left', fontsize=12)

# plt.tight_layout()
# plt.savefig('plots/strong_scaling_throughput.png', dpi=150, bbox_inches='tight')
# plt.close()
# print("Saved: plots/strong_scaling_throughput.png")

# ===========================================================================
# PLOT 4: Strong Scaling - Efficiency
# ===========================================================================
fig, ax = plt.subplots(figsize=(10, 7))

gpus = strong_scaling['num_gpus'].values
efficiency = strong_scaling['efficiency'].values

bars = ax.bar(gpus, efficiency, color=colors['strong'], width=0.6, edgecolor='black', linewidth=1.2)

# Add value labels on bars
for bar, eff in zip(bars, efficiency):
    height = bar.get_height()
    ax.annotate(f'{eff:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5), textcoords="offset points", ha='center', va='bottom',
                fontsize=12, fontweight='bold')

ax.axhline(y=100, color=colors['ideal'], linestyle='--', linewidth=2, label='Ideal Efficiency (100%)')
ax.set_xlabel('Number of GPUs', fontsize=14)
ax.set_ylabel('Efficiency (%)', fontsize=14)
ax.set_title('Strong Scaling: Parallelization Efficiency\n(Global Batch Size = 32)', fontsize=16, fontweight='bold')
ax.set_xticks(gpus)
ax.set_xticklabels([str(g) for g in gpus])
ax.set_ylim(0, 120)
ax.grid(True, alpha=0.3, axis='y')

# Vertical line: transition from 1 to 2 nodes (between 4 and 8 GPUs)
ax.axvline(x=6, color='gray', linestyle='--', linewidth=2, alpha=0.8, label='1→2 nodes')
ax.legend(loc='upper right', fontsize=12)

plt.tight_layout()
plt.savefig('plots/strong_scaling_efficiency.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plots/strong_scaling_efficiency.png")

# ===========================================================================
# WEAK SCALING ANALYSIS
# ===========================================================================
# Weak scaling: rows 4-8 (batch_size_per_gpu = 8, global_batch_size scales with GPUs)
weak_scaling = df.iloc[4:9].copy()

print("\nWeak Scaling Data:")
print(weak_scaling[['num_gpus', 'global_batch_size', 'batch_size_per_gpu', 'samples_per_sec', 'avg_epoch_time_sec']])

# For weak scaling, we measure how well throughput scales with problem size
# Ideal: throughput should scale linearly with GPUs (since work scales linearly)
base_throughput_weak = weak_scaling.iloc[0]['samples_per_sec']
weak_scaling['relative_speedup'] = weak_scaling['samples_per_sec'] / base_throughput_weak
weak_scaling['ideal_speedup'] = weak_scaling['num_gpus']
weak_scaling['efficiency'] = (weak_scaling['relative_speedup'] / weak_scaling['num_gpus']) * 100

colors['weak'] = '#A23B72'

# ===========================================================================
# PLOT 5: Weak Scaling - Speedup
# ===========================================================================
fig, ax = plt.subplots(figsize=(10, 7))

gpus = weak_scaling['num_gpus'].values
speedup = weak_scaling['relative_speedup'].values
ideal = weak_scaling['ideal_speedup'].values

ax.plot(gpus, ideal, 'o--', color=colors['ideal'], linewidth=2, markersize=10, label='Ideal Speedup')
ax.plot(gpus, speedup, 's-', color=colors['weak'], linewidth=2.5, markersize=12, label='Measured Speedup')

ax.set_xlabel('Number of GPUs', fontsize=14)
ax.set_ylabel('Speedup (relative to 1 GPU)', fontsize=14)
ax.set_title('Weak Scaling: Speedup vs Number of GPUs\n(Batch per GPU = 8)', fontsize=16, fontweight='bold')
ax.set_xticks(gpus)
ax.set_xticklabels([str(g) for g in gpus])
ax.legend(loc='upper left', fontsize=12)
ax.set_xlim(0, max(gpus) + 2)
ax.set_ylim(0, max(ideal) + 2)
ax.grid(True, alpha=0.3)

# Vertical lines: transitions between nodes
ax.axvline(x=6, color='gray', linestyle='--', linewidth=2, alpha=0.8, label='1→2 nodes')
ax.axvline(x=12, color='dimgray', linestyle='--', linewidth=2, alpha=0.8, label='2→4 nodes')
ax.legend(loc='upper left', fontsize=12)

plt.tight_layout()
plt.savefig('plots/weak_scaling_speedup.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plots/weak_scaling_speedup.png")

# ===========================================================================
# PLOT 6: Weak Scaling - Efficiency
# ===========================================================================
fig, ax = plt.subplots(figsize=(10, 7))

gpus = weak_scaling['num_gpus'].values
efficiency = weak_scaling['efficiency'].values

bars = ax.bar(range(len(gpus)), efficiency, color=colors['weak'], width=0.6, edgecolor='black', linewidth=1.2)

# Add value labels on bars
for bar, eff in zip(bars, efficiency):
    height = bar.get_height()
    ax.annotate(f'{eff:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5), textcoords="offset points", ha='center', va='bottom',
                fontsize=12, fontweight='bold')

ax.axhline(y=100, color=colors['ideal'], linestyle='--', linewidth=2, label='Ideal Efficiency (100%)')
ax.set_xlabel('Number of GPUs', fontsize=14)
ax.set_ylabel('Efficiency (%)', fontsize=14)
ax.set_title('Weak Scaling: Parallelization Efficiency\n(Batch per GPU = 8)', fontsize=16, fontweight='bold')
ax.set_xticks(range(len(gpus)))
ax.set_xticklabels([str(g) for g in gpus])
ax.set_ylim(0, 120)
ax.grid(True, alpha=0.3, axis='y')

# Vertical lines: transitions between nodes (bar chart uses indices 0-4)
# 1 GPU=idx0, 2 GPU=idx1, 4 GPU=idx2, 8 GPU=idx3, 16 GPU=idx4
ax.axvline(x=2.5, color='gray', linestyle='--', linewidth=2, alpha=0.8, label='1→2 nodes')
ax.axvline(x=3.5, color='dimgray', linestyle='--', linewidth=2, alpha=0.8, label='2→4 nodes')
ax.legend(loc='upper right', fontsize=12)

plt.tight_layout()
plt.savefig('plots/weak_scaling_efficiency.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plots/weak_scaling_efficiency.png")

# ===========================================================================
# PLOT 7: Weak Scaling - Time per Epoch
# ===========================================================================
fig, ax = plt.subplots(figsize=(10, 7))

gpus = weak_scaling['num_gpus'].values
epoch_time = weak_scaling['avg_epoch_time_sec'].values
# For weak scaling: dataset is fixed, global batch increases, so steps decrease proportionally
# Ideal time should decrease linearly: time_1gpu / num_gpus
ideal_time = weak_scaling.iloc[0]['avg_epoch_time_sec'] / gpus

ax.plot(gpus, ideal_time, 'o--', color=colors['ideal'], linewidth=2, markersize=10, label='Ideal Time')
ax.plot(gpus, epoch_time, 's-', color=colors['weak'], linewidth=2.5, markersize=12, label='Measured Time')

ax.set_xlabel('Number of GPUs', fontsize=14)
ax.set_ylabel('Time per Epoch (seconds)', fontsize=14)
ax.set_title('Weak Scaling: Time per Epoch vs Number of GPUs\n(Batch per GPU = 8)', fontsize=16, fontweight='bold')
ax.set_xticks(gpus)
ax.set_xticklabels([str(g) for g in gpus])
ax.legend(loc='upper right', fontsize=12)
ax.set_xlim(0, max(gpus) + 2)
ax.grid(True, alpha=0.3)

# Vertical lines: transitions between nodes
ax.axvline(x=6, color='gray', linestyle='--', linewidth=2, alpha=0.8, label='1→2 nodes')
ax.axvline(x=12, color='dimgray', linestyle='--', linewidth=2, alpha=0.8, label='2→4 nodes')
ax.legend(loc='upper right', fontsize=12)

plt.tight_layout()
plt.savefig('plots/weak_scaling_epoch_time.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plots/weak_scaling_epoch_time.png")

# ===========================================================================
# RESOURCE UTILIZATION ANALYSIS
# ===========================================================================

# ===========================================================================
# PLOT 8: GPU Utilization Comparison - Strong vs Weak Scaling
# ===========================================================================
fig, ax = plt.subplots(figsize=(10, 7))

ax.plot(strong_scaling['num_gpus'], strong_scaling['avg_gpu_utilization_pct'], 
        's-', color=colors['strong'], linewidth=2.5, markersize=12, label='Strong Scaling (GBS=32)')
ax.plot(weak_scaling['num_gpus'], weak_scaling['avg_gpu_utilization_pct'], 
        '^-', color=colors['weak'], linewidth=2.5, markersize=12, label='Weak Scaling (BpG=8)')

ax.set_xlabel('Number of GPUs', fontsize=14)
ax.set_ylabel('Average GPU Utilization (%)', fontsize=14)
ax.set_title('GPU Utilization vs Number of GPUs', fontsize=16, fontweight='bold')
ax.set_ylim(60, 100)
ax.grid(True, alpha=0.3)

# Vertical lines for node transitions
ax.axvline(x=6, color='gray', linestyle='--', linewidth=2, alpha=0.8, label='1→2 nodes')
ax.axvline(x=12, color='dimgray', linestyle='--', linewidth=2, alpha=0.8, label='2→4 nodes')
ax.legend(loc='lower left', fontsize=12)

plt.tight_layout()
plt.savefig('plots/gpu_utilization.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plots/gpu_utilization.png")

# ===========================================================================
# PLOT 9: GPU Memory Usage per GPU
# ===========================================================================
fig, ax = plt.subplots(figsize=(10, 7))

ax.plot(strong_scaling['num_gpus'], strong_scaling['avg_gpu_memory_mb'], 
        's-', color=colors['strong'], linewidth=2.5, markersize=12, label='Strong Scaling (GBS=32)')
ax.plot(weak_scaling['num_gpus'], weak_scaling['avg_gpu_memory_mb'], 
        '^-', color=colors['weak'], linewidth=2.5, markersize=12, label='Weak Scaling (BpG=8)')

ax.set_xlabel('Number of GPUs', fontsize=14)
ax.set_ylabel('Average GPU Memory per GPU (MB)', fontsize=14)
ax.set_title('GPU Memory Usage per GPU vs Number of GPUs', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)

# Vertical lines for node transitions
ax.axvline(x=6, color='gray', linestyle='--', linewidth=2, alpha=0.8, label='1→2 nodes')
ax.axvline(x=12, color='dimgray', linestyle='--', linewidth=2, alpha=0.8, label='2→4 nodes')
ax.legend(loc='upper right', fontsize=12)

plt.tight_layout()
plt.savefig('plots/gpu_memory.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plots/gpu_memory.png")

# ===========================================================================
# PLOT 11: Energy Efficiency (Samples/sec per Watt)
# ===========================================================================
fig, ax = plt.subplots(figsize=(10, 7))

# Calculate total power and efficiency
strong_scaling['total_power_w'] = strong_scaling['avg_gpu_power_w'] * strong_scaling['num_gpus']
strong_scaling['energy_efficiency'] = strong_scaling['samples_per_sec'] / strong_scaling['total_power_w']

weak_scaling['total_power_w'] = weak_scaling['avg_gpu_power_w'] * weak_scaling['num_gpus']
weak_scaling['energy_efficiency'] = weak_scaling['samples_per_sec'] / weak_scaling['total_power_w']

ax.plot(strong_scaling['num_gpus'], strong_scaling['energy_efficiency'], 
        's-', color=colors['strong'], linewidth=2.5, markersize=12, label='Strong Scaling (GBS=32)')
ax.plot(weak_scaling['num_gpus'], weak_scaling['energy_efficiency'], 
        '^-', color=colors['weak'], linewidth=2.5, markersize=12, label='Weak Scaling (BpG=8)')

ax.set_xlabel('Number of GPUs', fontsize=14)
ax.set_ylabel('Energy Efficiency (samples/sec/W)', fontsize=14)
ax.set_title('Energy Efficiency vs Number of GPUs', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)

# Vertical lines for node transitions
ax.axvline(x=6, color='gray', linestyle='--', linewidth=2, alpha=0.8, label='1→2 nodes')
ax.axvline(x=12, color='dimgray', linestyle='--', linewidth=2, alpha=0.8, label='2→4 nodes')
ax.legend(loc='upper left', fontsize=12)

plt.tight_layout()
plt.savefig('plots/energy_efficiency.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plots/energy_efficiency.png")

print("\n" + "="*50)
print("All plots have been generated successfully!")
print("Plots are saved in the 'plots/' subfolder.")
print("="*50)
