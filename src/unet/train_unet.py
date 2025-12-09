"""
DeepSpeed distributed training script for UNet segmentation model.
Adapted from train_detr_ddp.py for segmentation-specific training loop.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

import datetime
import time
import numpy as np
import deepspeed
from deepspeed.accelerator import get_accelerator

from unet_model import UNetSegmentor
from dataset_unet import SparkSegmentationDataset, collate_fn_segmentation

import warnings
warnings.filterwarnings("ignore", message=".*meta parameter.*")
import subprocess

# Segmentation classes
NUM_CLASSES = 3
CLASS_NAMES = ["background", "spacecraft_body", "solar_panels"]


def get_gpu_stats(device=None):
    """Get GPU utilization, memory usage and power draw.
    
    Uses nvidia-smi with specific GPU index based on CUDA_VISIBLE_DEVICES and LOCAL_RANK
    to query only the GPU actually used by this process.
    """
    try:
        # Get local rank to query the correct GPU
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Get the physical GPU index from CUDA_VISIBLE_DEVICES
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
                    'num_gpus_detected': 1
                }
    except Exception as e:
        pass
    return {
        'gpu_util_avg': 0,
        'gpu_mem_used_mb': 0,
        'gpu_mem_total_mb': 0,
        'gpu_power_w': 0,
        'num_gpus_detected': 0
    }


def compute_iou_multiclass(pred, target, num_classes=NUM_CLASSES, ignore_background=False):
    """
    Compute mean Intersection over Union (mIoU) for multi-class segmentation.
    
    Args:
        pred: Predicted class indices [B, H, W]
        target: Ground truth class indices [B, H, W]
        num_classes: Number of classes
        ignore_background: If True, exclude class 0 from mean
    
    Returns:
        mean IoU across classes, per-class IoU dict
    """
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
            ious[c] = float('nan')  # No pixels of this class
    
    # Mean IoU (excluding NaN values)
    valid_ious = [v for v in ious.values() if not np.isnan(v)]
    mean_iou = np.mean(valid_ious) if valid_ious else 0.0
    
    return mean_iou, ious


def compute_dice_multiclass(pred, target, num_classes=NUM_CLASSES, ignore_background=False):
    """
    Compute mean Dice coefficient for multi-class segmentation.
    
    Args:
        pred: Predicted class indices [B, H, W]
        target: Ground truth class indices [B, H, W]
        num_classes: Number of classes
        ignore_background: If True, exclude class 0 from mean
    
    Returns:
        mean Dice across classes, per-class Dice dict
    """
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


def train_model_unet(model_engine, train_loader, val_loader, train_sampler=None, num_epochs=100, 
                     validation=True, validate_every=1, save_path="model_weights", patience=15):
    """
    Training loop for UNet segmentation model with DeepSpeed.
    
    Args:
        train_sampler: DistributedSampler for setting epoch (proper shuffling)
        validate_every: Run validation only every N epochs (default=1, every epoch)
    """
    device = model_engine.device
    global_rank = dist.get_rank()
    
    if global_rank == 0:
        print(f'Device used: {device}')

    train_losses = []
    val_losses = []
    train_ious = []
    val_ious = []
    train_dices = []
    val_dices = []
    epoch_times = []
    learning_rates = []
    gpu_utilizations_all = []  # Collect multiple samples per epoch
    gpu_memory_usages_all = []
    gpu_power_draws_all = []
    training_start_time = datetime.datetime.now()
    
    # Early stopping
    best_val_loss = float('inf')
    best_val_iou = 0.0
    no_improve_epochs = 0
    
    if global_rank == 0:
        print("=" * 60)
        print("TRAINING STARTED (UNet Segmentation)")
        print("=" * 60)

    for epoch in range(num_epochs):
        # Set epoch for proper shuffling with DistributedSampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        if global_rank == 0:
            epoch_start_dt = datetime.datetime.now()
            print(f"\n[Rank 0] Starting epoch {epoch+1} at {epoch_start_dt.strftime('%Y-%m-%d %H:%M:%S.%f')}")
        
        epoch_start_time = time.time()
        model_engine.train()
        
        epoch_train_loss = 0.0
        epoch_train_iou = 0.0
        epoch_train_dice = 0.0
        batch_count = 0
        start_time_batch = time.time()
        
        # Timing accumulators
        time_data_loading = 0.0
        time_forward = 0.0
        time_backward = 0.0
        time_optimizer_step = 0.0
        time_sync = 0.0

        data_load_start = time.time()
        for batch_idx, batch in enumerate(train_loader):
            t0 = time.time()
            time_data_loading += t0 - data_load_start
            
            # Move data to device
            images = batch["images"].to(device)
            masks = batch["masks"].to(device)
            
            if batch_idx == 0 and epoch == 0 and global_rank == 0:
                print("\n--- Data Check ---")
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
                pred_classes = outputs['pred_masks']  # Already class indices from argmax
                iou, _ = compute_iou_multiclass(pred_classes, masks)
                dice, _ = compute_dice_multiclass(pred_classes, masks)
            
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
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model_engine.parameters(), max_norm=1.0)
            
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

            # Sample GPU stats more frequently for accurate benchmarking (every 5 batches)
            if batch_idx % 5 == 0 and global_rank == 0:
                gpu_stats = get_gpu_stats(device)
                gpu_utilizations_all.append(gpu_stats['gpu_util_avg'])
                gpu_memory_usages_all.append(gpu_stats['gpu_mem_used_mb'])
                gpu_power_draws_all.append(gpu_stats['gpu_power_w'])

            if batch_idx % 20 == 0 and global_rank == 0:
                batch_time = time.time() - start_time_batch
                print(f'[Epoch {epoch+1:3d}] Batch {batch_idx:4d} | Loss: {loss.item():.4f} | IoU: {iou:.4f} | Dice: {dice:.4f} | Time: {batch_time:.2f}s')
                start_time_batch = time.time()
            
            data_load_start = time.time()
        
        # Synchronization
        t_sync_start = time.time()
        dist.barrier()
        torch.cuda.synchronize()
        t_sync_end = time.time()
        time_sync = t_sync_end - t_sync_start
        
        train_time = time.time() - epoch_start_time

        avg_train_loss = epoch_train_loss / max(batch_count, 1)
        avg_train_iou = epoch_train_iou / max(batch_count, 1)
        avg_train_dice = epoch_train_dice / max(batch_count, 1)
        
        train_losses.append(avg_train_loss)
        train_ious.append(avg_train_iou)
        train_dices.append(avg_train_dice)

        # VALIDATION - only run every validate_every epochs
        avg_val_loss = 0.0
        avg_val_iou = 0.0
        avg_val_dice = 0.0
        val_time = 0.0
        run_validation_this_epoch = validation and ((epoch + 1) % validate_every == 0)
        
        if run_validation_this_epoch:
            val_start_time = time.time()
            model_engine.eval()
            
            epoch_val_loss = 0.0
            epoch_val_iou = 0.0
            epoch_val_dice = 0.0
            val_batch_count = 0
            
            with torch.no_grad():
                for val_batch_idx, batch in enumerate(val_loader):
                    images = batch["images"].to(device)
                    masks = batch["masks"].to(device)
                    
                    if model_engine.fp16_enabled():
                        images = images.half()
                    
                    outputs = model_engine(images, masks)
                    loss = outputs['loss']
                    
                    pred_classes = outputs['pred_masks']
                    iou, _ = compute_iou_multiclass(pred_classes, masks)
                    dice, _ = compute_dice_multiclass(pred_classes, masks)
                    
                    epoch_val_loss += loss.item()
                    epoch_val_iou += iou
                    epoch_val_dice += dice
                    val_batch_count += 1
            
            torch.cuda.synchronize()
            val_time = time.time() - val_start_time
            
            # Aggregate validation metrics across all ranks
            val_metrics_tensor = torch.tensor([epoch_val_loss, epoch_val_iou, epoch_val_dice, val_batch_count], device=device)
            dist.all_reduce(val_metrics_tensor, op=dist.ReduceOp.SUM)
            
            total_val_loss = val_metrics_tensor[0].item()
            total_val_iou = val_metrics_tensor[1].item()
            total_val_dice = val_metrics_tensor[2].item()
            total_val_batches = val_metrics_tensor[3].item()
            
            avg_val_loss = total_val_loss / max(total_val_batches, 1)
            avg_val_iou = total_val_iou / max(total_val_batches, 1)
            avg_val_dice = total_val_dice / max(total_val_batches, 1)
            
            val_losses.append(avg_val_loss)
            val_ious.append(avg_val_iou)
            val_dices.append(avg_val_dice)

        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        current_lr = model_engine.optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        if global_rank == 0:
            print("\n" + "=" * 60)
            print(f"EPOCH {epoch+1}/{num_epochs} SUMMARY")
            print("=" * 60)
            print(f"Total Epoch Time: {epoch_time:.2f}s | LR: {current_lr:.2e}")
            print("-" * 60)
            print("TIMING BREAKDOWN:")
            print(f"  Training Total:     {train_time:.2f}s")
            print(f"    - Data Loading:   {time_data_loading:.2f}s ({100*time_data_loading/train_time:.1f}%)")
            print(f"    - Forward Pass:   {time_forward:.2f}s ({100*time_forward/train_time:.1f}%)")
            print(f"    - Backward Pass:  {time_backward:.2f}s ({100*time_backward/train_time:.1f}%)")
            print(f"    - Optimizer Step: {time_optimizer_step:.2f}s ({100*time_optimizer_step/train_time:.1f}%)")
            print(f"    - Sync Barrier:   {time_sync:.2f}s ({100*time_sync/train_time:.1f}%)")
            if run_validation_this_epoch:
                print(f"  Validation:         {val_time:.2f}s")
            else:
                print(f"  Validation:         (skipped, every {validate_every} epochs)")
            print("-" * 60)
            print(f"Train Loss: {avg_train_loss:.4f} | IoU: {avg_train_iou:.4f} | Dice: {avg_train_dice:.4f}")
            if run_validation_this_epoch:
                print(f"Val Loss: {avg_val_loss:.4f} | IoU: {avg_val_iou:.4f} | Dice: {avg_val_dice:.4f}")
            print("=" * 60 + "\n")
        
        # Early stopping based on validation loss
        if run_validation_this_epoch:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_iou = avg_val_iou
                no_improve_epochs = 0
                
                # Save best model
                if global_rank == 0:
                    save_start = time.time()
                    os.makedirs(save_path, exist_ok=True)
                    base_model = model_engine.module if hasattr(model_engine, 'module') else model_engine
                    base_model.save_pretrained(os.path.join(save_path, "unet_best"))
                    save_time = time.time() - save_start
                    print(f"✓ Best model saved to {save_path}/unet_best (save time: {save_time:.2f}s)")
            else:
                no_improve_epochs += 1
                if global_rank == 0:
                    print(f"✗ No improvement for {no_improve_epochs} epoch(s)")
            
            if no_improve_epochs >= patience:
                if global_rank == 0:
                    print(f"\n🛑 Early stopping at epoch {epoch+1}: no improvement for {patience} epochs.")
                break
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0 and global_rank == 0:
            save_start = time.time()
            os.makedirs(save_path, exist_ok=True)
            base_model = model_engine.module if hasattr(model_engine, 'module') else model_engine
            base_model.save_pretrained(os.path.join(save_path, f"unet_epoch_{epoch+1}"))
            save_time = time.time() - save_start
            print(f"📁 Checkpoint saved at epoch {epoch+1} (save time: {save_time:.2f}s)")

    # Save final model
    if global_rank == 0:
        save_start = time.time()
        os.makedirs(save_path, exist_ok=True)
        base_model = model_engine.module if hasattr(model_engine, 'module') else model_engine
        base_model.save_pretrained(os.path.join(save_path, "unet_final"))
        save_time = time.time() - save_start
        print(f"\n✅ Final model saved to {save_path}/unet_final (save time: {save_time:.2f}s)")
        
        # Save loss and metrics plots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Loss plot
        # Create x-axis for train epochs and validation epochs (validation happens every `validate_every` epochs)
        train_epochs = list(range(1, len(train_losses) + 1))
        axes[0].plot(train_epochs, train_losses, label="Train Loss", linewidth=2)
        if val_losses:
            # Validation was run at epochs validate_every, 2*validate_every, ...
            val_epochs = list(range(validate_every, len(train_losses) + 1, validate_every))
            # Ensure lengths match in case of early stopping or mismatches
            val_epochs = val_epochs[:len(val_losses)]
            axes[0].plot(val_epochs, val_losses, label="Val Loss", linewidth=2, marker='o')
        axes[0].set_xlabel("Epoch", fontsize=12)
        axes[0].set_ylabel("Loss", fontsize=12)
        axes[0].set_title("Loss", fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # IoU plot
        axes[1].plot(train_ious, label="Train IoU", linewidth=2)
        if val_ious:
            axes[1].plot(val_ious, label="Val IoU", linewidth=2)
        axes[1].set_xlabel("Epoch", fontsize=12)
        axes[1].set_ylabel("IoU", fontsize=12)
        axes[1].set_title("Intersection over Union", fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        # Dice plot
        axes[2].plot(train_dices, label="Train Dice", linewidth=2)
        if val_dices:
            axes[2].plot(val_dices, label="Val Dice", linewidth=2)
        axes[2].set_xlabel("Epoch", fontsize=12)
        axes[2].set_ylabel("Dice", fontsize=12)
        axes[2].set_title("Dice Coefficient", fontsize=14, fontweight='bold')
        axes[2].legend(fontsize=11)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(save_path, "training_plots.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"📈 Training plots saved to {plot_path}")

    training_stats = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_ious': train_ious,
        'val_ious': val_ious,
        'train_dices': train_dices,
        'val_dices': val_dices,
        'epoch_times': epoch_times,
        'learning_rates': learning_rates,
        'gpu_utilizations': gpu_utilizations_all,
        'gpu_memory_usages': gpu_memory_usages_all,
        'gpu_power_draws': gpu_power_draws_all,
        'training_start_time': training_start_time,
        'training_end_time': datetime.datetime.now(),
        'best_val_loss': best_val_loss,
        'best_val_iou': best_val_iou
    }
    
    return model_engine, training_stats


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--deepspeed', action='store_true')
    parser.add_argument('--benchmark', action='store_true', help='Enable benchmark mode')
    args = parser.parse_args()
    
    deepspeed.init_distributed()
    
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    # ============================================================================
    # CONFIGURATION - Load and potentially modify DeepSpeed config
    # ============================================================================
    DS_CONFIG_PATH = "src/unet/ds_config_unet.json"
    
    # Load DeepSpeed config
    with open(DS_CONFIG_PATH, 'r') as f:
        ds_config = json.load(f)
    
    # Override batch size from environment if provided (for benchmarks)
    if 'BATCH_SIZE' in os.environ:
        BATCH_SIZE = int(os.environ['BATCH_SIZE'])
        ds_config['train_micro_batch_size_per_gpu'] = BATCH_SIZE
        if global_rank == 0:
            print(f"Overriding batch size from env: {BATCH_SIZE}")
    else:
        BATCH_SIZE = ds_config.get('train_micro_batch_size_per_gpu', 8)
    
    # Override epochs from environment if provided
    N_EPOCHS = int(os.environ.get('N_EPOCHS', 30))
    
    # Update scheduler total_num_steps based on dataset size and epochs
    # (will be updated after dataset loading)
    
    DATA_ROOT = "/project/scratch/p200981/spark2024"
    TRAIN_CSV = f"{DATA_ROOT}/train.csv"
    VAL_CSV = f"{DATA_ROOT}/val.csv"
    IMAGE_ROOT = f"{DATA_ROOT}/images"
    MASK_ROOT = f"{DATA_ROOT}/mask"
    
    # Training settings (can be overridden by env vars for benchmarking)
    # Enable downsampling from env var or --benchmark flag
    DOWN_SAMPLE = args.benchmark or os.environ.get('DOWNSAMPLE', '0') == '1'
    DOWN_SAMPLE_SUBSET = 1000  # Number of samples when downsampling is enabled
    TARGET_SIZE = (512, 512)  # Reduced from 1024 for faster training
    VALIDATION = True
    VALIDATE_EVERY = 2  # Run validation every N epochs
    PATIENCE = 15
    SAVE_PATH = "model_weights_unet"
    NUM_WORKERS = 7  # DataLoader workers
    
    # Model settings
    N_CHANNELS = 3  # RGB images
    N_CLASSES = 3   # 3-class segmentation: background, spacecraft body, solar panels
    USE_TINY_MODEL = True   # Use tiny UNet (16 base, 3 levels) - much faster

    # ============================================================================
    # LOAD MODEL
    # ============================================================================
    if global_rank == 0:
        print("Loading UNet model...")
    
    model = UNetSegmentor(
        n_channels=N_CHANNELS,
        n_classes=N_CLASSES,
        base_filters=64,
        bilinear=True,
        use_tiny=USE_TINY_MODEL
    )
    
    if global_rank == 0:
        print("Model loaded successfully!")

    # ============================================================================
    # PREPARE DATASETS
    # ============================================================================
    if global_rank == 0:
        print("Preparing datasets...")

    train_dataset = SparkSegmentationDataset(
        csv_path=TRAIN_CSV,
        image_root=IMAGE_ROOT,
        mask_root=MASK_ROOT,
        split="train",
        target_size=TARGET_SIZE,
        augment=True  # Enable augmentation for training
    )

    val_dataset = SparkSegmentationDataset(
        csv_path=VAL_CSV,
        image_root=IMAGE_ROOT,
        mask_root=MASK_ROOT,
        split="val",
        target_size=TARGET_SIZE,
        augment=False
    )

    if DOWN_SAMPLE:
        train_dataset = Subset(train_dataset, range(min(DOWN_SAMPLE_SUBSET, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(DOWN_SAMPLE_SUBSET // 10, len(val_dataset))))

    if global_rank == 0:
        print(f"Data prepared: train samples = {len(train_dataset)}, val samples = {len(val_dataset)}")
    
    # Update DeepSpeed config scheduler based on actual dataset size
    steps_per_epoch = len(train_dataset) // (BATCH_SIZE * world_size)
    total_steps = steps_per_epoch * N_EPOCHS
    ds_config['scheduler']['params']['total_num_steps'] = total_steps
    
    if global_rank == 0:
        print(f"⚙️  Updated scheduler total_num_steps to {total_steps} ({steps_per_epoch} steps/epoch × {N_EPOCHS} epochs)")

    # Create validation loader with DistributedSampler
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=dist.get_world_size(),
        rank=global_rank,
        shuffle=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        sampler=val_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn_segmentation
    )

    # ============================================================================
    # MODEL SUMMARY
    # ============================================================================
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    if global_rank == 0:
        print("\n" + "=" * 60)
        print("MODEL SUMMARY")
        print("=" * 60)
        print(f"Model type: UNetSegmentor")
        print(f"Input channels: {N_CHANNELS}")
        print(f"Output classes: {N_CLASSES}")
        print(f"Image size: {TARGET_SIZE}")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print("=" * 60 + "\n")
        print("\nInitializing DeepSpeed...")
    
    # Create train dataloader manually for full control
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=global_rank,
        shuffle=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # DistributedSampler handles shuffling
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn_segmentation
    )
    
    # ============================================================================
    # DEEPSPEED INITIALIZATION
    # ============================================================================
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=parameters,
        config=ds_config  # Use modified config dict instead of file path
    )
    
    if global_rank == 0:
        print("DeepSpeed initialization completed.\n")
        print(f"Effective batch size per GPU: {BATCH_SIZE}")
        print(f"Global batch size: {BATCH_SIZE * world_size}")
        print(f"Gradient accumulation steps: {ds_config.get('gradient_accumulation_steps', 1)}")
        print(f"Type of train_loader: {type(train_loader)}\n")
    
    # ============================================================================
    # TRAINING
    # ============================================================================
    training_start = datetime.datetime.now()
    
    if global_rank == 0 and args.benchmark:
        print("\n" + "=" * 60)
        print("BENCHMARK MODE")
        print("=" * 60)
        print(f"World size: {world_size} GPUs")
        print(f"Batch per GPU: {BATCH_SIZE}")
        print(f"Global batch size: {BATCH_SIZE * world_size}")
        print(f"Epochs: {N_EPOCHS}")
        print(f"Dataset samples: {len(train_dataset)}")
        print("=" * 60 + "\n")
    
    model, training_stats = train_model_unet(
        model_engine, 
        train_loader, 
        val_loader,
        train_sampler=train_sampler,
        num_epochs=N_EPOCHS, 
        validation=VALIDATION,
        validate_every=VALIDATE_EVERY,
        save_path=SAVE_PATH,
        patience=PATIENCE
    )

    if global_rank == 0:
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        if args.benchmark:
            total_time = (training_stats['training_end_time'] - training_stats['training_start_time']).total_seconds()
            avg_epoch_time = np.mean(training_stats['epoch_times'])
            print("\nBENCHMARK RESULTS:")
            print(f"Total time: {total_time:.2f}s")
            print(f"Avg epoch time: {avg_epoch_time:.2f}s")
            print(f"Throughput: {len(train_dataset) / avg_epoch_time:.2f} samples/sec/epoch")

    # ============================================================================
    # SAVE TRAINING REPORT (CSV)
    # ============================================================================
    if global_rank == 0:
        import csv
        report_dir = "benchmark_results"
        os.makedirs(report_dir, exist_ok=True) 
        csv_file = os.path.join(report_dir, "training_summary.csv")
        
        # Check if file exists to determine if we need to write header
        file_exists = os.path.isfile(csv_file)
        
        total_duration = (training_stats['training_end_time'] - training_stats['training_start_time']).total_seconds()
        
        # Aggregate metrics
        summary = {
            'timestamp': training_start.strftime('%Y%m%d_%H%M%S'),
            'start_time': training_stats['training_start_time'].strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': training_stats['training_end_time'].strftime('%Y-%m-%d %H:%M:%S'),
            'total_duration_sec': total_duration,
            'num_epochs': len(training_stats['train_losses']),
            'num_gpus': world_size,
            'batch_size_per_gpu': BATCH_SIZE,
            'global_batch_size': BATCH_SIZE * world_size,
            'gradient_accumulation_steps': ds_config.get('gradient_accumulation_steps', 1),
            'fp16_enabled': ds_config.get('fp16', {}).get('enabled', False),
            'zero_optimization_stage': ds_config.get('zero_optimization', {}).get('stage', 0),
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'image_size': TARGET_SIZE,
            'num_classes': N_CLASSES,
            'tiny_model': USE_TINY_MODEL,
            
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
            
            # Average metrics
            'avg_train_loss': np.mean(training_stats['train_losses']),
            'avg_train_iou': np.mean(training_stats['train_ious']),
            'avg_train_dice': np.mean(training_stats['train_dices']),
            'avg_val_loss': np.mean(training_stats['val_losses']) if training_stats['val_losses'] else None,
            'avg_val_iou': np.mean(training_stats['val_ious']) if training_stats['val_ious'] else None,
            'avg_val_dice': np.mean(training_stats['val_dices']) if training_stats['val_dices'] else None,
            
            # Timing metrics
            'avg_epoch_time_sec': np.mean(training_stats['epoch_times']),
            'min_epoch_time_sec': np.min(training_stats['epoch_times']),
            'max_epoch_time_sec': np.max(training_stats['epoch_times']),
            'total_epoch_time_sec': np.sum(training_stats['epoch_times']),
            'samples_per_sec': len(train_dataset) / np.mean(training_stats['epoch_times']),
            
            # Learning rate
            'initial_lr': training_stats['learning_rates'][0],
            'final_lr': training_stats['learning_rates'][-1],
            
            # GPU metrics (averaged across epochs)
            'avg_gpu_utilization_pct': np.mean(training_stats['gpu_utilizations']) if training_stats['gpu_utilizations'] else None,
            'max_gpu_utilization_pct': np.max(training_stats['gpu_utilizations']) if training_stats['gpu_utilizations'] else None,
            'avg_gpu_memory_mb': np.mean(training_stats['gpu_memory_usages']) if training_stats['gpu_memory_usages'] else None,
            'max_gpu_memory_mb': np.max(training_stats['gpu_memory_usages']) if training_stats['gpu_memory_usages'] else None,
            'avg_gpu_power_w': np.mean(training_stats['gpu_power_draws']) if training_stats['gpu_power_draws'] else None,
            'max_gpu_power_w': np.max(training_stats['gpu_power_draws']) if training_stats['gpu_power_draws'] else None,
        }
        
        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summary.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(summary)
        
        print(f"\n📊 Training summary appended to: {csv_file}\n")
