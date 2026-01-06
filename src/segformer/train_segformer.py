"""
DeepSpeed distributed training script for SegFormer segmentation model.
Supports multi-node multi-GPU training with full configuration via YAML.

Usage:
    # Single node (4 GPUs)
    deepspeed --num_gpus=4 train_segformer.py --config config_segformer.yaml
    
    # Multi-node via SLURM (see segformer_train.sh)
    srun python -u train_segformer.py --config config_segformer.yaml

Features:
    - Full YAML configuration for all hyperparameters
    - DeepSpeed ZeRO optimization for memory efficiency
    - Mixed precision training (FP16/BF16)
    - Learning rate scheduling with warmup
    - Early stopping with patience
    - Automatic training parameter logging for reproducibility
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import os
import sys
import argparse
import yaml
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime
import time
import numpy as np
import deepspeed
from deepspeed.accelerator import get_accelerator
import subprocess
import warnings
warnings.filterwarnings("ignore", message=".*meta parameter.*")

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'unet'))
from dataset_unet import SparkSegmentationDataset, collate_fn_segmentation

from segformer_model import SegFormerSegmentor, list_available_variants


# =============================================================================
# CONFIGURATION LOADING
# =============================================================================

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_config_with_args(config: dict, args: argparse.Namespace) -> dict:
    """Merge command line arguments with config file (CLI takes precedence)."""
    # Override with command line arguments if provided (only CLI args, no env vars)
    if args.batch_size is not None:
        config['training']['batch_size'] = int(args.batch_size)
    if args.epochs is not None:
        config['training']['epochs'] = int(args.epochs)
    if args.variant is not None:
        config['model']['variant'] = str(args.variant)
    if args.image_size is not None:
        config['training']['image_size'] = int(args.image_size)
    if args.lr is not None:
        config['training']['optimizer']['lr'] = float(args.lr)
    if args.save_path is not None:
        config['output']['save_path'] = str(args.save_path)
    if args.debug:
        config['experiment']['debug_mode'] = True
        
    return config


def save_training_params(config: dict, save_path: str, world_size: int):
    """Save all training parameters to a file for reproducibility."""
    os.makedirs(save_path, exist_ok=True)
    
    # Add runtime info
    params = {
        'config': config,
        'runtime': {
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'world_size': world_size,
            'cuda_version': torch.version.cuda,
            'pytorch_version': torch.__version__,
            'hostname': os.environ.get('HOSTNAME', 'unknown'),
            'slurm_job_id': os.environ.get('SLURM_JOB_ID', 'N/A'),
            'slurm_num_nodes': os.environ.get('SLURM_JOB_NUM_NODES', '1'),
        }
    }
    
    # Save as YAML
    yaml_path = os.path.join(save_path, 'training_params.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(params, f, default_flow_style=False, sort_keys=False)
    
    # Also save as JSON for easy parsing
    json_path = os.path.join(save_path, 'training_params.json')
    with open(json_path, 'w') as f:
        json.dump(params, f, indent=2)
    
    print(f"[Config] Training parameters saved to {yaml_path}")
    return params


# =============================================================================
# DEEPSPEED CONFIG GENERATION
# =============================================================================

def generate_deepspeed_config(config: dict, total_steps: int) -> dict:
    """Generate DeepSpeed configuration from YAML config."""
    training_cfg = config['training']
    ds_cfg = config.get('deepspeed', {})
    opt_cfg = training_cfg.get('optimizer', {})
    sched_cfg = training_cfg.get('scheduler', {})
    
    # Warmup steps calculation
    warmup_epochs = float(sched_cfg.get('warmup_epochs', 5))
    if warmup_epochs < 1:
        warmup_steps = int(total_steps * warmup_epochs)
    else:
        steps_per_epoch = total_steps // int(training_cfg.get('epochs', 100))
        warmup_steps = int(warmup_epochs * steps_per_epoch)
    
    # FP16/BF16 configuration
    precision = str(training_cfg.get('precision', 'fp16'))
    fp16_config = {
        "enabled": precision == 'fp16',
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    }
    bf16_config = {
        "enabled": precision == 'bf16'
    }
    
    # ZeRO configuration
    zero_stage = int(ds_cfg.get('zero_stage', 1))
    zero_config = {
        "stage": zero_stage,
        "allgather_partitions": True,
        "allgather_bucket_size": int(ds_cfg.get('allgather_bucket_size', 5e8)),
        "reduce_scatter": True,
        "reduce_bucket_size": int(ds_cfg.get('reduce_bucket_size', 5e8)),
        "overlap_comm": bool(ds_cfg.get('overlap_comm', True)),
        "contiguous_gradients": True
    }
    
    # Offload config for ZeRO-2/3
    if ds_cfg.get('offload_optimizer', False):
        zero_config["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
    if zero_stage == 3 and ds_cfg.get('offload_params', False):
        zero_config["offload_param"] = {"device": "cpu", "pin_memory": True}
    
    # Scheduler configuration - ensure all values are proper types
    scheduler_type = str(sched_cfg.get('type', 'warmup_cosine'))
    if scheduler_type in ['warmup_cosine', 'cosine']:
        ds_scheduler = {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": float(sched_cfg.get('warmup_lr', 1e-7)),
                "warmup_max_lr": float(opt_cfg.get('lr', 6e-5)),
                "warmup_num_steps": int(warmup_steps),
                "total_num_steps": int(total_steps)
            }
        }
    elif scheduler_type == 'polynomial':
        ds_scheduler = {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": float(sched_cfg.get('warmup_lr', 1e-7)),
                "warmup_max_lr": float(opt_cfg.get('lr', 6e-5)),
                "warmup_num_steps": int(warmup_steps)
            }
        }
    else:
        ds_scheduler = {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": float(sched_cfg.get('warmup_lr', 1e-7)),
                "warmup_max_lr": float(opt_cfg.get('lr', 6e-5)),
                "warmup_num_steps": int(warmup_steps),
                "total_num_steps": int(total_steps)
            }
        }
    
    ds_config = {
        "train_micro_batch_size_per_gpu": int(training_cfg.get('batch_size', 8)),
        "gradient_accumulation_steps": int(training_cfg.get('gradient_accumulation_steps', 1)),
        "fp16": fp16_config,
        "bf16": bf16_config,
        "zero_optimization": zero_config,
        "optimizer": {
            "type": str(opt_cfg.get('type', 'AdamW')),
            "params": {
                "lr": float(opt_cfg.get('lr', 6e-5)),
                "betas": [float(b) for b in opt_cfg.get('betas', [0.9, 0.999])],
                "eps": float(opt_cfg.get('eps', 1e-8)),
                "weight_decay": float(opt_cfg.get('weight_decay', 0.01))
            }
        },
        "scheduler": ds_scheduler,
        "gradient_clipping": float(training_cfg.get('gradient_clip', 1.0)),
        "wall_clock_breakdown": False
    }
    
    return ds_config


# =============================================================================
# GPU MONITORING
# =============================================================================

def get_gpu_stats():
    """Get GPU utilization, memory usage and power draw."""
    try:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_visible:
            gpu_indices = cuda_visible.split(',')
            physical_gpu = gpu_indices[local_rank] if local_rank < len(gpu_indices) else gpu_indices[0]
        else:
            physical_gpu = str(local_rank)
        
        result = subprocess.run(
            ['nvidia-smi', f'--id={physical_gpu}',
             '--query-gpu=utilization.gpu,memory.used,memory.total,power.draw', 
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=1
        )
        
        if result.returncode == 0:
            parts = result.stdout.strip().split(',')
            if len(parts) >= 4:
                return {
                    'gpu_util_avg': float(parts[0].strip()),
                    'gpu_mem_used_mb': float(parts[1].strip()),
                    'gpu_mem_total_mb': float(parts[2].strip()),
                    'gpu_power_w': float(parts[3].strip()),
                }
    except Exception:
        pass
    return {'gpu_util_avg': 0, 'gpu_mem_used_mb': 0, 'gpu_mem_total_mb': 0, 'gpu_power_w': 0}


# =============================================================================
# METRICS
# =============================================================================

def compute_iou_multiclass(pred, target, num_classes=3, ignore_background=False):
    """Compute mean Intersection over Union (mIoU) for multi-class segmentation."""
    ious = {}
    start_class = 1 if ignore_background else 0
    
    for c in range(start_class, num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum() - intersection
        
        if union > 0:
            ious[c] = (intersection / union).item()
        else:
            ious[c] = float('nan')
    
    valid_ious = [v for v in ious.values() if not np.isnan(v)]
    mean_iou = np.mean(valid_ious) if valid_ious else 0.0
    
    return mean_iou, ious


def compute_dice_multiclass(pred, target, num_classes=3, ignore_background=False):
    """Compute mean Dice coefficient for multi-class segmentation."""
    dices = {}
    start_class = 1 if ignore_background else 0
    
    for c in range(start_class, num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        
        intersection = (pred_c * target_c).sum()
        total = pred_c.sum() + target_c.sum()
        
        if total > 0:
            dices[c] = (2. * intersection / total).item()
        else:
            dices[c] = float('nan')
    
    valid_dices = [v for v in dices.values() if not np.isnan(v)]
    mean_dice = np.mean(valid_dices) if valid_dices else 0.0
    
    return mean_dice, dices


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_model(
    model_engine, 
    train_loader, 
    val_loader, 
    config: dict,
    train_sampler=None
):
    """
    Training loop for SegFormer with DeepSpeed.
    """
    training_cfg = config['training']
    output_cfg = config['output']
    
    num_epochs = training_cfg['epochs']
    patience = training_cfg['patience']
    validate_every = training_cfg.get('validate_every', 1)
    save_path = output_cfg['save_path']
    log_every = output_cfg.get('log_every', 20)
    checkpoint_every = output_cfg.get('checkpoint_every', 10)
    num_classes = config['data']['num_classes']
    
    device = model_engine.device
    global_rank = dist.get_rank()
    
    # Training stats
    train_losses, val_losses = [], []
    train_ious, val_ious = [], []
    train_dices, val_dices = [], []
    epoch_times = []
    learning_rates = []
    gpu_stats_all = []
    
    training_start_time = datetime.datetime.now()
    best_val_loss = float('inf')
    best_val_iou = 0.0
    no_improve_epochs = 0
    
    if global_rank == 0:
        print("\n" + "=" * 70)
        print("TRAINING STARTED (SegFormer Segmentation)")
        print("=" * 70)

    for epoch in range(num_epochs):
        # Set epoch for proper shuffling
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        epoch_start_time = time.time()
        model_engine.train()
        
        epoch_train_loss = 0.0
        epoch_train_iou = 0.0
        epoch_train_dice = 0.0
        batch_count = 0
        
        # Timing accumulators
        time_data_loading = 0.0
        time_forward = 0.0
        time_backward = 0.0
        time_optimizer_step = 0.0
        
        data_load_start = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            t0 = time.time()
            time_data_loading += t0 - data_load_start
            
            # Move data to device
            images = batch["images"].to(device)
            masks = batch["masks"].to(device)
            
            if batch_idx == 0 and epoch == 0 and global_rank == 0:
                print(f"\n--- Data Check ---")
                print(f"Images shape: {images.shape}")
                print(f"Masks shape: {masks.shape}")
                print(f"Masks unique values: {torch.unique(masks)}")
                print("------------------\n")
            
            if model_engine.fp16_enabled():
                images = images.half()
            
            # Forward pass
            t1 = time.time()
            outputs = model_engine(images, masks)
            torch.cuda.synchronize()
            t2 = time.time()
            time_forward += t2 - t1
            
            loss = outputs['loss']
            
            # Compute metrics
            with torch.no_grad():
                pred_classes = outputs['pred_masks']
                iou, _ = compute_iou_multiclass(pred_classes, masks, num_classes)
                dice, _ = compute_dice_multiclass(pred_classes, masks, num_classes)
            
            if torch.isnan(loss) or torch.isinf(loss):
                if global_rank == 0:
                    print(f"WARNING: Invalid loss at batch {batch_idx} - skipping")
                data_load_start = time.time()
                continue
            
            # Backward pass
            t3 = time.time()
            model_engine.backward(loss)
            torch.cuda.synchronize()
            t4 = time.time()
            time_backward += t4 - t3
            
            # Gradient clipping (DeepSpeed handles this via config, but explicit for safety)
            torch.nn.utils.clip_grad_norm_(
                model_engine.parameters(), 
                max_norm=training_cfg.get('gradient_clip', 1.0)
            )
            
            # Optimizer step
            t5 = time.time()
            model_engine.step()
            torch.cuda.synchronize()
            t6 = time.time()
            time_optimizer_step += t6 - t5
            
            epoch_train_loss += loss.item()
            epoch_train_iou += iou
            epoch_train_dice += dice
            batch_count += 1
            
            # Sample GPU stats
            if batch_idx % 5 == 0 and global_rank == 0:
                gpu_stats_all.append(get_gpu_stats())
            
            # Log progress
            if batch_idx % log_every == 0 and global_rank == 0:
                print(f'[Epoch {epoch+1:3d}] Batch {batch_idx:4d} | '
                      f'Loss: {loss.item():.4f} | IoU: {iou:.4f} | Dice: {dice:.4f}')
            
            data_load_start = time.time()
        
        # Synchronization
        dist.barrier()
        torch.cuda.synchronize()
        
        train_time = time.time() - epoch_start_time
        
        avg_train_loss = epoch_train_loss / max(batch_count, 1)
        avg_train_iou = epoch_train_iou / max(batch_count, 1)
        avg_train_dice = epoch_train_dice / max(batch_count, 1)
        
        train_losses.append(avg_train_loss)
        train_ious.append(avg_train_iou)
        train_dices.append(avg_train_dice)
        
        # VALIDATION
        avg_val_loss = 0.0
        avg_val_iou = 0.0
        avg_val_dice = 0.0
        val_time = 0.0
        run_validation = (epoch + 1) % validate_every == 0
        
        if run_validation:
            val_start_time = time.time()
            model_engine.eval()
            
            epoch_val_loss = 0.0
            epoch_val_iou = 0.0
            epoch_val_dice = 0.0
            val_batch_count = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    images = batch["images"].to(device)
                    masks = batch["masks"].to(device)
                    
                    if model_engine.fp16_enabled():
                        images = images.half()
                    
                    outputs = model_engine(images, masks)
                    loss = outputs['loss']
                    
                    pred_classes = outputs['pred_masks']
                    iou, _ = compute_iou_multiclass(pred_classes, masks, num_classes)
                    dice, _ = compute_dice_multiclass(pred_classes, masks, num_classes)
                    
                    epoch_val_loss += loss.item()
                    epoch_val_iou += iou
                    epoch_val_dice += dice
                    val_batch_count += 1
            
            torch.cuda.synchronize()
            val_time = time.time() - val_start_time
            
            # Aggregate validation metrics across ranks
            val_metrics = torch.tensor(
                [epoch_val_loss, epoch_val_iou, epoch_val_dice, val_batch_count], 
                device=device
            )
            dist.all_reduce(val_metrics, op=dist.ReduceOp.SUM)
            
            total_val_batches = val_metrics[3].item()
            avg_val_loss = val_metrics[0].item() / max(total_val_batches, 1)
            avg_val_iou = val_metrics[1].item() / max(total_val_batches, 1)
            avg_val_dice = val_metrics[2].item() / max(total_val_batches, 1)
            
            val_losses.append(avg_val_loss)
            val_ious.append(avg_val_iou)
            val_dices.append(avg_val_dice)
        
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        current_lr = model_engine.optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # Print epoch summary
        if global_rank == 0:
            print("\n" + "=" * 70)
            print(f"EPOCH {epoch+1}/{num_epochs} SUMMARY")
            print("=" * 70)
            print(f"Total Time: {epoch_time:.2f}s | LR: {current_lr:.2e}")
            print("-" * 70)
            print("TIMING BREAKDOWN:")
            print(f"  Training Total:     {train_time:.2f}s")
            print(f"    - Data Loading:   {time_data_loading:.2f}s ({100*time_data_loading/train_time:.1f}%)")
            print(f"    - Forward Pass:   {time_forward:.2f}s ({100*time_forward/train_time:.1f}%)")
            print(f"    - Backward Pass:  {time_backward:.2f}s ({100*time_backward/train_time:.1f}%)")
            print(f"    - Optimizer Step: {time_optimizer_step:.2f}s ({100*time_optimizer_step/train_time:.1f}%)")
            if run_validation:
                print(f"  Validation:         {val_time:.2f}s")
            print("-" * 70)
            print(f"Train | Loss: {avg_train_loss:.4f} | IoU: {avg_train_iou:.4f} | Dice: {avg_train_dice:.4f}")
            if run_validation:
                print(f"Val   | Loss: {avg_val_loss:.4f} | IoU: {avg_val_iou:.4f} | Dice: {avg_val_dice:.4f}")
            print("=" * 70 + "\n")
        
        # Early stopping and model saving
        if run_validation:
            improved = False
            if avg_val_iou > best_val_iou:
                best_val_iou = avg_val_iou
                improved = True
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                improved = True
            
            if improved:
                no_improve_epochs = 0
                if global_rank == 0:
                    os.makedirs(save_path, exist_ok=True)
                    base_model = model_engine.module if hasattr(model_engine, 'module') else model_engine
                    base_model.save_pretrained(os.path.join(save_path, "segformer_best"))
                    print(f"✓ Best model saved to {save_path}/segformer_best")
            else:
                no_improve_epochs += 1
                if global_rank == 0:
                    print(f"✗ No improvement for {no_improve_epochs} epoch(s)")
            
            if no_improve_epochs >= patience:
                if global_rank == 0:
                    print(f"\n🛑 Early stopping at epoch {epoch+1}: no improvement for {patience} epochs.")
                break
        
        # Periodic checkpoint
        if (epoch + 1) % checkpoint_every == 0 and global_rank == 0:
            os.makedirs(save_path, exist_ok=True)
            base_model = model_engine.module if hasattr(model_engine, 'module') else model_engine
            base_model.save_pretrained(os.path.join(save_path, f"segformer_epoch_{epoch+1}"))
            print(f"📁 Checkpoint saved at epoch {epoch+1}")
    
    # Save final model
    if global_rank == 0:
        os.makedirs(save_path, exist_ok=True)
        base_model = model_engine.module if hasattr(model_engine, 'module') else model_engine
        base_model.save_pretrained(os.path.join(save_path, "segformer_final"))
        print(f"\n✅ Final model saved to {save_path}/segformer_final")
        
        # Save training plots
        save_training_plots(
            train_losses, val_losses, train_ious, val_ious, 
            train_dices, val_dices, validate_every, save_path
        )
    
    training_stats = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_ious': train_ious,
        'val_ious': val_ious,
        'train_dices': train_dices,
        'val_dices': val_dices,
        'epoch_times': epoch_times,
        'learning_rates': learning_rates,
        'gpu_stats': gpu_stats_all,
        'training_start_time': training_start_time,
        'training_end_time': datetime.datetime.now(),
        'best_val_loss': best_val_loss,
        'best_val_iou': best_val_iou
    }
    
    return model_engine, training_stats


def save_training_plots(train_losses, val_losses, train_ious, val_ious, 
                        train_dices, val_dices, validate_every, save_path):
    """Save training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    train_epochs = list(range(1, len(train_losses) + 1))
    val_epochs = list(range(validate_every, len(train_losses) + 1, validate_every))[:len(val_losses)]
    
    # Loss
    axes[0].plot(train_epochs, train_losses, label="Train", linewidth=2)
    if val_losses:
        axes[0].plot(val_epochs, val_losses, label="Val", linewidth=2, marker='o')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss", fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # IoU
    axes[1].plot(train_epochs, train_ious, label="Train", linewidth=2)
    if val_ious:
        axes[1].plot(val_epochs, val_ious, label="Val", linewidth=2, marker='o')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("IoU")
    axes[1].set_title("Mean IoU", fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Dice
    axes[2].plot(train_epochs, train_dices, label="Train", linewidth=2)
    if val_dices:
        axes[2].plot(val_epochs, val_dices, label="Val", linewidth=2, marker='o')
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Dice")
    axes[2].set_title("Mean Dice", fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "training_plots.png"), dpi=150)
    plt.close()
    print(f"📈 Training plots saved to {save_path}/training_plots.png")


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="SegFormer Training with DeepSpeed")
    
    # Config file
    parser.add_argument('--config', type=str, default='src/segformer/config_segformer.yaml',
                        help='Path to configuration YAML file')
    
    # CLI overrides
    parser.add_argument('--variant', type=str, choices=['b0', 'b1', 'b2', 'b3', 'b4', 'b5'],
                        help='SegFormer variant (overrides config)')
    parser.add_argument('--image_size', type=int, help='Image size (overrides config)')
    parser.add_argument('--batch_size', type=int, help='Batch size per GPU (overrides config)')
    parser.add_argument('--epochs', type=int, help='Number of epochs (overrides config)')
    parser.add_argument('--lr', type=float, help='Learning rate (overrides config)')
    parser.add_argument('--save_path', type=str, help='Output directory (overrides config)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with small dataset')
    
    # DeepSpeed
    parser.add_argument('--local_rank', type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize distributed
    deepspeed.init_distributed()
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Load and merge config
    config = load_config(args.config)
    config = merge_config_with_args(config, args)
    
    # Extract config sections
    data_cfg = config['data']
    model_cfg = config['model']
    training_cfg = config['training']
    aug_cfg = config['augmentation']
    output_cfg = config['output']
    exp_cfg = config['experiment']
    hw_cfg = config.get('hardware', {})
    
    # Set seeds for reproducibility
    seed = exp_cfg.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Hardware optimizations
    if hw_cfg.get('cudnn_benchmark', True):
        torch.backends.cudnn.benchmark = True
    if hw_cfg.get('allow_tf32', True) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    if global_rank == 0:
        print("\n" + "=" * 70)
        print("SEGFORMER TRAINING")
        print("=" * 70)
        print(f"Config file: {args.config}")
        print(f"Model variant: {model_cfg['variant']}")
        print(f"Image size: {training_cfg['image_size']}")
        print(f"Batch size per GPU: {training_cfg['batch_size']}")
        print(f"World size: {world_size} GPUs")
        print(f"Effective batch size: {training_cfg['batch_size'] * world_size}")
        print("=" * 70)
    
    # ==========================================================================
    # LOAD MODEL
    # ==========================================================================
    if global_rank == 0:
        print("\nLoading SegFormer model...")
        print(f"Variant: {model_cfg['variant']}")
        print(f"Pretrained: {model_cfg['pretrained']}")
    
    loss_config = training_cfg.get('loss', {})
    
    model = SegFormerSegmentor(
        n_classes=data_cfg['num_classes'],
        variant=model_cfg['variant'],
        pretrained=model_cfg['pretrained'],
        freeze_backbone=model_cfg.get('freeze_backbone', False),
        dropout=model_cfg.get('dropout', 0.1),
        loss_config=loss_config
    )
    
    # ==========================================================================
    # PREPARE DATASETS
    # ==========================================================================
    if global_rank == 0:
        print("\nPreparing datasets...")
    
    data_root = data_cfg['data_root']
    target_size = (training_cfg['image_size'], training_cfg['image_size'])
    
    train_dataset = SparkSegmentationDataset(
        csv_path=os.path.join(data_root, data_cfg['train_csv']),
        image_root=os.path.join(data_root, data_cfg['image_root']),
        mask_root=os.path.join(data_root, data_cfg['mask_root']),
        split="train",
        target_size=target_size,
        augment=aug_cfg.get('enabled', True)
    )
    
    val_dataset = SparkSegmentationDataset(
        csv_path=os.path.join(data_root, data_cfg['val_csv']),
        image_root=os.path.join(data_root, data_cfg['image_root']),
        mask_root=os.path.join(data_root, data_cfg['mask_root']),
        split="val",
        target_size=target_size,
        augment=False
    )
    
    # Debug mode - subsample dataset
    if exp_cfg.get('debug_mode', False):
        debug_samples = exp_cfg.get('debug_samples', 100)
        train_indices = list(range(min(debug_samples, len(train_dataset))))
        val_indices = list(range(min(debug_samples // 2, len(val_dataset))))
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
        if global_rank == 0:
            print(f"[Debug Mode] Using {len(train_dataset)} train, {len(val_dataset)} val samples")
    
    if global_rank == 0:
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
    
    # ==========================================================================
    # DEEPSPEED SETUP
    # ==========================================================================
    batch_size = training_cfg['batch_size']
    num_workers = training_cfg.get('num_workers', 8)
    
    # Calculate total training steps
    steps_per_epoch = len(train_dataset) // (batch_size * world_size)
    total_steps = steps_per_epoch * training_cfg['epochs']
    
    # Generate DeepSpeed config
    ds_config = generate_deepspeed_config(config, total_steps)
    
    if global_rank == 0:
        print(f"\nDeepSpeed config:")
        print(f"  ZeRO stage: {ds_config['zero_optimization']['stage']}")
        print(f"  FP16: {ds_config['fp16']['enabled']}")
        print(f"  BF16: {ds_config['bf16']['enabled']}")
        print(f"  Total steps: {total_steps}")
    
    # Create samplers and dataloaders
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=global_rank, shuffle=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn_segmentation
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn_segmentation
    )
    
    # Initialize DeepSpeed
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=parameters,
        config=ds_config
    )
    
    if global_rank == 0:
        print("\nDeepSpeed initialized successfully!")
        
        # Save training parameters for reproducibility
        save_training_params(config, output_cfg['save_path'], world_size)
    
    # ==========================================================================
    # TRAINING
    # ==========================================================================
    model_engine, training_stats = train_model(
        model_engine,
        train_loader,
        val_loader,
        config,
        train_sampler=train_sampler
    )
    
    # ==========================================================================
    # SAVE FINAL REPORT
    # ==========================================================================
    if global_rank == 0:
        save_training_report(config, training_stats, world_size, output_cfg['save_path'])
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"Best validation IoU: {training_stats['best_val_iou']:.4f}")
        print(f"Best validation loss: {training_stats['best_val_loss']:.4f}")
        print(f"Model saved to: {output_cfg['save_path']}")
        print("=" * 70)


def save_training_report(config, training_stats, world_size, save_path):
    """Save comprehensive training report."""
    import csv
    
    training_cfg = config['training']
    model_cfg = config['model']
    
    report_dir = "benchmark_results"
    os.makedirs(report_dir, exist_ok=True)
    csv_file = os.path.join(report_dir, "training_summary.csv")
    
    file_exists = os.path.isfile(csv_file)
    
    total_duration = (training_stats['training_end_time'] - training_stats['training_start_time']).total_seconds()
    
    summary = {
        'timestamp': training_stats['training_start_time'].strftime('%Y%m%d_%H%M%S'),
        'model': f"SegFormer-{model_cfg['variant']}",
        'start_time': training_stats['training_start_time'].strftime('%Y-%m-%d %H:%M:%S'),
        'end_time': training_stats['training_end_time'].strftime('%Y-%m-%d %H:%M:%S'),
        'total_duration_sec': total_duration,
        'num_epochs': len(training_stats['train_losses']),
        'num_nodes': int(os.environ.get('SLURM_JOB_NUM_NODES', 1)),
        'num_gpus': world_size,
        'batch_size_per_gpu': training_cfg['batch_size'],
        'global_batch_size': training_cfg['batch_size'] * world_size,
        'image_size': training_cfg['image_size'],
        'num_classes': config['data']['num_classes'],
        'pretrained': model_cfg['pretrained'],
        
        # Final metrics
        'final_train_loss': training_stats['train_losses'][-1],
        'final_train_iou': training_stats['train_ious'][-1],
        'final_train_dice': training_stats['train_dices'][-1],
        'final_val_loss': training_stats['val_losses'][-1] if training_stats['val_losses'] else None,
        'final_val_iou': training_stats['val_ious'][-1] if training_stats['val_ious'] else None,
        'final_val_dice': training_stats['val_dices'][-1] if training_stats['val_dices'] else None,
        
        # Best metrics
        'best_val_loss': training_stats['best_val_loss'],
        'best_val_iou': training_stats['best_val_iou'],
        
        # Timing
        'avg_epoch_time_sec': np.mean(training_stats['epoch_times']),
        'initial_lr': training_stats['learning_rates'][0],
        'final_lr': training_stats['learning_rates'][-1],
    }
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(summary)
    
    print(f"📊 Training summary appended to: {csv_file}")


if __name__ == "__main__":
    main()
