"""
DeepSpeed distributed training script for DETR model.
Adapted from train_ddp.py for DETR-specific training loop.
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

from models.detr_model import DETRDetector
from utils.spark_detection_dataset_detr import SparkDetectionDatasetDETR, collate_fn_detr

import warnings
warnings.filterwarnings("ignore", message=".*meta parameter.*")


def train_model_detr(model_engine, train_loader, val_loader, num_epochs=100, validation=False, 
                     save_path="model_weights", patience=15):
    """
    Training loop for DETR model with DeepSpeed.
    
    DETR computes its own loss internally (classification + bbox + GIoU losses).
    """
    device = model_engine.device
    global_rank = dist.get_rank()
    
    if global_rank == 0:
        print(f'Device used: {device}')

    train_losses = []
    val_losses = []
    epoch_times = []
    learning_rates = []
    training_start_time = datetime.datetime.now()
    
    # Early stopping
    best_val_loss = float('inf')
    no_improve_epochs = 0
    
    if global_rank == 0:
        print("=" * 60)
        print("TRAINING STARTED (DETR)")
        print("=" * 60)

    for epoch in range(num_epochs):
        if global_rank == 0:
            epoch_start_dt = datetime.datetime.now()
            print(f"\n[Rank 0] Starting epoch {epoch+1} at {epoch_start_dt.strftime('%Y-%m-%d %H:%M:%S.%f')}")
        epoch_start_time = time.time()
        model_engine.train()
        epoch_train_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_bbox_loss = 0.0
        epoch_giou_loss = 0.0
        batch_count = 0
        start_time_batch = time.time()
        
        # Timing accumulators for training steps
        time_data_loading = 0.0
        time_forward = 0.0
        time_backward = 0.0
        time_optimizer_step = 0.0
        time_sync = 0.0

        data_load_start = time.time()
        for batch_idx, batch in enumerate(train_loader):
            # Data loading time (includes previous iteration's data prefetch)
            t0 = time.time()
            time_data_loading += t0 - data_load_start
            
            # Data transfer to device
            pixel_values = batch["pixel_values"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
            
            if batch_idx == 0 and epoch == 0 and global_rank == 0:
                print("\n--- Data Check ---")
                print(f"Pixel values shape: {pixel_values.shape}")
                print(f"Number of targets: {len(labels)}")
                print(f"First target keys: {labels[0].keys()}")
                print(f"First target class_labels: {labels[0]['class_labels']}")
                print(f"First target boxes shape: {labels[0]['boxes'].shape}")
                print("------------------\n")
            
            if model_engine.fp16_enabled():
                pixel_values = pixel_values.half()
            
            # Forward pass timing
            t1 = time.time()
            outputs = model_engine(pixel_values=pixel_values, labels=labels)
            torch.cuda.synchronize()  # Ensure forward is complete
            t2 = time.time()
            time_forward += t2 - t1
            
            loss = outputs.loss
            
            # Extract individual loss components if available
            if hasattr(outputs, 'loss_dict'):
                loss_dict = outputs.loss_dict
                epoch_cls_loss += loss_dict.get('loss_ce', torch.tensor(0)).item()
                epoch_bbox_loss += loss_dict.get('loss_bbox', torch.tensor(0)).item()
                epoch_giou_loss += loss_dict.get('loss_giou', torch.tensor(0)).item()
            
            if torch.isnan(loss) or torch.isinf(loss):
                if global_rank == 0:
                    print(f"WARNING: Invalid loss at batch {batch_idx} - skipping")
                data_load_start = time.time()
                continue
            
            # Backward pass timing
            t3 = time.time()
            model_engine.backward(loss)
            torch.cuda.synchronize()  # Ensure backward is complete
            t4 = time.time()
            time_backward += t4 - t3
            
            # Gradient clipping (important for DETR stability)
            torch.nn.utils.clip_grad_norm_(model_engine.parameters(), max_norm=0.1)
            
            # Optimizer step timing
            t5 = time.time()
            model_engine.step()
            torch.cuda.synchronize()  # Ensure step is complete
            t6 = time.time()
            time_optimizer_step += t6 - t5
            
            epoch_train_loss += loss.item()
            batch_count += 1

            if batch_idx % 40 == 0 and global_rank == 0:
                batch_time = time.time() - start_time_batch
                print(f'[Epoch {epoch+1:3d}] Batch {batch_idx:4d} | Loss: {loss.item():.4f} | Time: {batch_time:.2f}s')
                start_time_batch = time.time()
            
            data_load_start = time.time()
        
        # Synchronization timing after training loop
        t_sync_start = time.time()
        dist.barrier()
        torch.cuda.synchronize()
        t_sync_end = time.time()
        time_sync = t_sync_end - t_sync_start
        
        train_time = time.time() - epoch_start_time

        avg_train_loss = epoch_train_loss / max(batch_count, 1)
        avg_cls_loss = epoch_cls_loss / max(batch_count, 1)
        avg_bbox_loss = epoch_bbox_loss / max(batch_count, 1)
        avg_giou_loss = epoch_giou_loss / max(batch_count, 1)
        train_losses.append(avg_train_loss)

        # VALIDATION
        avg_val_loss = 0.0
        val_time = 0.0
        if validation:
            val_start_time = time.time()
            val_batch_start_time = time.time()
            model_engine.eval()
            epoch_val_loss = 0.0
            val_batch_count = 0
            with torch.no_grad():
                for val_batch_idx, batch in enumerate(val_loader):
                    pixel_values = batch["pixel_values"].to(device)
                    labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
                    
                    if model_engine.fp16_enabled():
                        pixel_values = pixel_values.half()
                    
                    outputs = model_engine(pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss
                    
                    epoch_val_loss += loss.item()
                    val_batch_count += 1
                    
                    if global_rank == 0:
                        # if val_batch_idx % 40 == 0 and global_rank == 0:
                        val_batch_time = time.time() - val_batch_start_time
                        print(f'[Epoch {epoch+1:3d}] Val Batch {val_batch_idx:4d} | Loss: {loss.item():.4f} | Time: {val_batch_time:.2f}s')
                        val_batch_start_time = time.time()
            
            torch.cuda.synchronize()
            val_time = time.time() - val_start_time
            
            # Aggregate validation loss across all ranks
            val_loss_tensor = torch.tensor([epoch_val_loss, val_batch_count], device=device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            total_val_loss = val_loss_tensor[0].item()
            total_val_batches = val_loss_tensor[1].item()
            avg_val_loss = total_val_loss / max(total_val_batches, 1)
            val_losses.append(avg_val_loss)

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
            if validation:
                print(f"  Validation:         {val_time:.2f}s")
            print("-" * 60)
            print(f"Train Loss: {avg_train_loss:.4f} | CE: {avg_cls_loss:.4f} | BBox: {avg_bbox_loss:.4f} | GIoU: {avg_giou_loss:.4f}")
            if validation:
                print(f"Val Loss: {avg_val_loss:.4f}")
            print("=" * 60 + "\n")
        
        # Early stopping (only on validation loss)
        save_time = 0.0
        if validation:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                no_improve_epochs = 0
                
                # Save best model
                if global_rank == 0:
                    save_start = time.time()
                    os.makedirs(save_path, exist_ok=True)
                    # Get base model for saving
                    base_model = model_engine.module if hasattr(model_engine, 'module') else model_engine
                    base_model.save_pretrained(os.path.join(save_path, "detr_best"))
                    save_time = time.time() - save_start
                    print(f"✓ Best model saved to {save_path}/detr_best (save time: {save_time:.2f}s)")
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
            base_model.save_pretrained(os.path.join(save_path, f"detr_epoch_{epoch+1}"))
            save_time = time.time() - save_start
            print(f"📁 Checkpoint saved at epoch {epoch+1} (save time: {save_time:.2f}s)")

    # Save final model
    if global_rank == 0:
        save_start = time.time()
        os.makedirs(save_path, exist_ok=True)
        base_model = model_engine.module if hasattr(model_engine, 'module') else model_engine
        base_model.save_pretrained(os.path.join(save_path, "detr_final"))
        save_time = time.time() - save_start
        print(f"\n✅ Final model saved to {save_path}/detr_final (save time: {save_time:.2f}s)")
        
        # Save loss plots
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Training Loss", linewidth=2)
        if val_losses:
            plt.plot(val_losses, label="Validation Loss", linewidth=2)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("Training and Validation Loss", fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(save_path, "loss_plot.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"📈 Loss plot saved to {plot_path}")

    training_stats = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'epoch_times': epoch_times,
        'learning_rates': learning_rates,
        'training_start_time': training_start_time,
        'training_end_time': datetime.datetime.now()
    }
    
    return model_engine, training_stats


if __name__ == "__main__":
    deepspeed.init_distributed()
    
    global_rank = dist.get_rank()

    # ============================================================================
    # CONFIGURATION
    # ============================================================================
    DATA_ROOT = "/project/scratch/p200981/spark2024"
    TRAIN_CSV = f"{DATA_ROOT}/train.csv"
    VAL_CSV = f"{DATA_ROOT}/val.csv"
    IMAGE_ROOT = f"{DATA_ROOT}/images"
    DOWN_SAMPLE = False
    DOWN_SAMPLE_SUBSET = 100
    BATCH_SIZE = 64
    N_EPOCHS = 2
    LEARNING_RATE = 1e-4      # Lower LR for fine-tuning pretrained model
    BACKBONE_LR = 1e-5        # Even lower LR for backbone
    VALIDATION = True
    PATIENCE = 15
    SAVE_PATH = "model_weights"

    # ============================================================================
    # LOAD CLASS MAPPINGS FROM CSV
    # ============================================================================
    train_df = pd.read_csv(TRAIN_CSV)
    class_names = sorted(train_df["Class"].unique())
    id2label = {i: name for i, name in enumerate(class_names)}
    label2id = {name: i for i, name in enumerate(class_names)}
    num_classes = len(class_names)
    
    if global_rank == 0:
        print(f"Number of classes: {num_classes}")
        print(f"id2label: {id2label}")

    # ============================================================================
    # LOAD MODEL
    # ============================================================================
    if global_rank == 0:
        print("Loading DETR model...")
    
    model = DETRDetector(
        num_classes=num_classes,
        model_name="facebook/detr-resnet-50",
        id2label=id2label,
        label2id=label2id
    )
    
    if global_rank == 0:
        print("Model loaded successfully!")

    # ============================================================================
    # PREPARE DATASETS
    # ============================================================================
    if global_rank == 0:
        print("Preparing datasets...")

    train_dataset = SparkDetectionDatasetDETR(
        csv_path=TRAIN_CSV,
        image_root=IMAGE_ROOT,
        split="train",
        image_processor=model.image_processor
    )

    val_dataset = SparkDetectionDatasetDETR(
        csv_path=VAL_CSV,
        image_root=IMAGE_ROOT,
        split="val",
        image_processor=model.image_processor
    )

    if DOWN_SAMPLE:
        train_dataset = Subset(train_dataset, range(min(DOWN_SAMPLE_SUBSET, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(DOWN_SAMPLE_SUBSET // 10, len(val_dataset))))

    if global_rank == 0:
        print(f"Data prepared: train samples = {len(train_dataset)}, val samples = {len(val_dataset)}")

    # Create validation loader with DistributedSampler to split across GPUs
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
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn_detr
    )

    # ============================================================================
    # MODEL SUMMARY
    # ============================================================================
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    if global_rank == 0:
        print("\n" + "=" * 60)
        print("MODEL SUMMARY")
        print("=" * 60)
        print(f"Model type: DETRDetector")
        print(f"Base model: facebook/detr-resnet-50")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print("=" * 60 + "\n")
        print("\nInitializing DeepSpeed...")
    
    # ============================================================================
    # DEEPSPEED INITIALIZATION
    # ============================================================================
    # Note: DeepSpeed will create its own DataLoader for training
    model_engine, optimizer, train_loader, _ = deepspeed.initialize(
        model=model,
        model_parameters=parameters,
        training_data=train_dataset,
        config="scripts/ds_config_detr.json",
        collate_fn=collate_fn_detr
    )
    
    if global_rank == 0:
        print("DeepSpeed initialization completed.\n")
        print(f"Type of train_loader: {type(train_loader)}")
    
    # ============================================================================
    # TRAINING
    # ============================================================================
    training_start = datetime.datetime.now()
    model, training_stats = train_model_detr(
        model_engine, 
        train_loader, 
        val_loader, 
        num_epochs=N_EPOCHS, 
        validation=VALIDATION,
        save_path=SAVE_PATH,
        patience=PATIENCE
    )

    if global_rank == 0:
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 60)

    # ============================================================================
    # SAVE TRAINING REPORT
    # ============================================================================
    if global_rank == 0:
        report_dir = "training_reports"
        os.makedirs(report_dir, exist_ok=True) 
        report_file = f"training_report_detr_{training_start.strftime('%Y%m%d_%H%M%S')}.txt"
        with open(os.path.join(report_dir, report_file), 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("TRAINING REPORT - DETR Detector Model\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("GENERAL INFORMATION\n")
            f.write("-" * 70 + "\n")
            f.write(f"Training Start: {training_stats['training_start_time'].strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Training End:   {training_stats['training_end_time'].strftime('%Y-%m-%d %H:%M:%S')}\n")
            total_duration = (training_stats['training_end_time'] - training_stats['training_start_time']).total_seconds()
            f.write(f"Total Duration: {total_duration/3600:.2f} hours ({total_duration/60:.2f} minutes)\n")
            f.write(f"Number of Epochs: {len(training_stats['train_losses'])}\n")
            f.write(f"Batch Size: {BATCH_SIZE}\n")
            f.write(f"Train Samples: {len(train_dataset)}\n")
            f.write(f"Val Samples: {len(val_dataset)}\n\n")
            
            f.write("MODEL CONFIGURATION\n")
            f.write("-" * 70 + "\n")
            f.write(f"Model: DETRDetector (facebook/detr-resnet-50)\n")
            f.write(f"Number of Classes: {num_classes}\n")
            f.write(f"Learning Rate: {LEARNING_RATE}\n")
            f.write(f"Backbone LR: {BACKBONE_LR}\n")
            f.write(f"Device: {model_engine.device}\n\n")
            
            f.write("TRAINING PERFORMANCE\n")
            f.write("-" * 70 + "\n")
            f.write(f"Final Train Loss: {training_stats['train_losses'][-1]:.6f}\n")
            if VALIDATION and training_stats['val_losses']:
                f.write(f"Final Val Loss:   {training_stats['val_losses'][-1]:.6f}\n")
            f.write(f"Best Train Loss:  {min(training_stats['train_losses']):.6f} (Epoch {training_stats['train_losses'].index(min(training_stats['train_losses']))+1})\n")
            if VALIDATION and training_stats['val_losses']:
                f.write(f"Best Val Loss:    {min(training_stats['val_losses']):.6f} (Epoch {training_stats['val_losses'].index(min(training_stats['val_losses']))+1})\n")
            f.write(f"Average Epoch Time: {np.mean(training_stats['epoch_times']):.2f}s\n")
            f.write(f"Final Learning Rate: {training_stats['learning_rates'][-1]:.2e}\n\n")
            
            f.write("EPOCH DETAILS\n")
            f.write("-" * 70 + "\n")
            if VALIDATION:
                f.write(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12} | {'LR':>12} | {'Time (s)':>10}\n")
                f.write("-" * 70 + "\n")
                for i in range(len(training_stats['train_losses'])):
                    val_loss_str = f"{training_stats['val_losses'][i]:12.6f}" if i < len(training_stats['val_losses']) else "N/A".rjust(12)
                    f.write(f"{i+1:6d} | {training_stats['train_losses'][i]:12.6f} | {val_loss_str} | "
                           f"{training_stats['learning_rates'][i]:12.2e} | {training_stats['epoch_times'][i]:10.2f}\n")
            else:
                f.write(f"{'Epoch':>6} | {'Train Loss':>12} | {'LR':>12} | {'Time (s)':>10}\n")
                f.write("-" * 70 + "\n")
                for i in range(len(training_stats['train_losses'])):
                    f.write(f"{i+1:6d} | {training_stats['train_losses'][i]:12.6f} | "
                           f"{training_stats['learning_rates'][i]:12.2e} | {training_stats['epoch_times'][i]:10.2f}\n")
            
            f.write("\n" + "=" * 70 + "\n")
        
        print(f"\nTraining report saved to: {report_file}\n")
        
        np.savez(os.path.join(report_dir, f"training_data_detr_{training_start.strftime('%Y%m%d_%H%M%S')}.npz"),
                 train_losses=np.array(training_stats['train_losses']),
                 val_losses=np.array(training_stats['val_losses']),
                 epoch_times=np.array(training_stats['epoch_times']),
                 learning_rates=np.array(training_stats['learning_rates']))
        print(f"Training data saved to: training_data_detr_{training_start.strftime('%Y%m%d_%H%M%S')}.npz\n")
