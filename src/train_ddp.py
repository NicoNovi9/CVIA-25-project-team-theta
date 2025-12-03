import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as T
from torch.utils.data import Subset
import torch.distributed as dist
import os

import datetime
import time
import numpy as np
import deepspeed
from deepspeed.accelerator import get_accelerator

from models.simple_model import SimpleDetector
from models.yolo_model import YOLODetector
from utils.spark_detection_dataset import SparkDetectionDataset

def train_model(model_engine, train_loader, val_loader, num_epochs=100, validation=False):
    device = model_engine.device
    global_rank = dist.get_rank()
    
    if global_rank == 0:
        print(f'Device used: {device}')

    ce_loss = nn.CrossEntropyLoss()
    bbox_loss_fn = nn.SmoothL1Loss()
    
    optimizer = model_engine.optimizer

    train_losses = []
    val_losses = []
    epoch_times = []
    learning_rates = []
    training_start_time = datetime.datetime.now()
    
    if global_rank == 0:
        print("=" * 60)
        print("TRAINING STARTED")
        print("=" * 60)

    for epoch in range(num_epochs):
        start_time = time.time()
        model_engine.train()
        epoch_train_loss = 0.0
        batch_count = 0
        start_time_batch = time.time()

        for batch_idx, (imgs, bboxes, labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            bboxes = bboxes.to(device)
            labels = labels.to(device)
            
            if batch_idx == 0 and epoch == 0 and global_rank == 0:
                print("\n--- Data Check ---")
                print(f"Image shape: {imgs.shape} | BBox shape: {bboxes.shape} | Labels shape: {labels.shape}")
                print(f"Image range: [{imgs.min():.3f}, {imgs.max():.3f}] | Mean: {imgs.mean():.3f}")
                print(f"Label classes: {torch.unique(labels).tolist()}")
                print("------------------\n")
            
            if model_engine.fp16_enabled():
                imgs = imgs.half()
            
            logits, pred_bbox = model_engine(imgs)
            
            if logits.dtype == torch.float16:
                logits = logits.float()
            if pred_bbox.dtype == torch.float16:
                pred_bbox = pred_bbox.float()
            
            loss_cls = ce_loss(logits, labels)
            loss_bbox = bbox_loss_fn(pred_bbox, bboxes)
            loss = loss_cls + loss_bbox
            
            if torch.isnan(loss) or torch.isinf(loss):
                if global_rank == 0:
                    print(f"WARNING: Invalid loss at batch {batch_idx} - skipping")
                continue
            
            model_engine.backward(loss)
            model_engine.step()
            epoch_train_loss += loss.item()
            batch_count += 1

            if batch_idx % 10 == 0 and global_rank == 0:
                batch_time = time.time() - start_time_batch
                print(f'[Epoch {epoch+1:3d}] Batch {batch_idx:4d} | Loss: {loss.item():.4f} | Cls: {loss_cls.item():.4f} | BBox: {loss_bbox.item():.4f} | Time: {batch_time:.2f}s')
                start_time_batch = time.time()

        avg_train_loss = epoch_train_loss / batch_count
        train_losses.append(avg_train_loss)

        # VALIDATION
        if validation:
            model_engine.eval()
            epoch_val_loss = 0.0
            val_batch_count = 0
            with torch.no_grad():
                for imgs, bboxes, labels in val_loader:
                    imgs = imgs.to(device)
                    bboxes = bboxes.to(device)
                    labels = labels.to(device)
                    
                    if model_engine.fp16_enabled():
                        imgs = imgs.half()
                    
                    logits, pred_bbox = model_engine(imgs)
                    
                    if logits.dtype == torch.float16:
                        logits = logits.float()
                    if pred_bbox.dtype == torch.float16:
                        pred_bbox = pred_bbox.float()
                    
                    loss_cls = ce_loss(logits, labels)
                    loss_bbox = bbox_loss_fn(pred_bbox, bboxes)
                    loss = loss_cls + loss_bbox
                    
                    epoch_val_loss += loss.item()
                    val_batch_count += 1

            avg_val_loss = epoch_val_loss / val_batch_count
            val_losses.append(avg_val_loss)

        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        if global_rank == 0:
            print("\n" + "=" * 60)
            print(f"EPOCH {epoch+1}/{num_epochs} SUMMARY")
            print("=" * 60)
            print(f"Duration: {epoch_time:.2f}s | LR: {optimizer.param_groups[0]['lr']:.2e}")
            if validation:
                print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            else:
                print(f"Train Loss: {avg_train_loss:.4f}")
            print("=" * 60 + "\n")

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

    DATA_ROOT = "/project/scratch/p200981/spark2024"
    DOWN_SAMPLE = False
    DOWN_SAMPLE_SUBSET = 10
    BATCH_SIZE = 512
    N_EPOCHS = 100
    LEARNING_RATE = 1e-3
    VALIDATION = False

    transform = T.Compose([
        #T.Resize((256, 256)),
        T.ToTensor(),
    ])

    if global_rank == 0:
        print("Preparing datasets")

    train_dataset = SparkDetectionDataset(
        csv_path=f"{DATA_ROOT}/train.csv",
        image_root=f"{DATA_ROOT}/images",
        split="train",
        transform=transform
    )

    val_dataset = SparkDetectionDataset(
        csv_path=f"{DATA_ROOT}/val.csv",
        image_root=f"{DATA_ROOT}/images",
        split="val",
        transform=transform
    )

    if DOWN_SAMPLE:
        train_dataset = Subset(train_dataset, range(DOWN_SAMPLE_SUBSET))
        val_dataset = Subset(val_dataset, range(DOWN_SAMPLE_SUBSET // 10))

    if global_rank == 0:
        print(f"Data prepared: train samples = {len(train_dataset)}, val samples = {len(val_dataset)}")

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = YOLODetector(num_classes=10)
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    if global_rank == 0:
        print("\n" + "=" * 60)
        print("MODEL SUMMARY")
        print("=" * 60)
        print(f"Model type: {type(model).__name__}")
        print(f"Model class: {model.__class__.__module__}.{model.__class__.__name__}")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print("=" * 60 + "\n")
        print("\nInitializing DeepSpeed...")
    
    model_engine, optimizer, train_loader, _ = deepspeed.initialize(
        model=model,
        model_parameters=parameters,
        training_data=train_dataset,
        config="scripts/ds_config.json"
    )
    
    if global_rank == 0:
        print("DeepSpeed initialization completed.\n")
        print(f"Type of train_loader: {type(train_loader)}")
    
    training_start = datetime.datetime.now()
    model, training_stats = train_model(model_engine, train_loader, val_loader, num_epochs=N_EPOCHS, validation=VALIDATION)

    if global_rank == 0:
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        report_file = f"training_report_{training_start.strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("TRAINING REPORT - Simple Detector Model\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("GENERAL INFORMATION\n")
            f.write("-" * 70 + "\n")
            f.write(f"Training Start: {training_stats['training_start_time'].strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Training End:   {training_stats['training_end_time'].strftime('%Y-%m-%d %H:%M:%S')}\n")
            total_duration = (training_stats['training_end_time'] - training_stats['training_start_time']).total_seconds()
            f.write(f"Total Duration: {total_duration/3600:.2f} hours ({total_duration/60:.2f} minutes)\n")
            f.write(f"Number of Epochs: {N_EPOCHS}\n")
            f.write(f"Batch Size: {BATCH_SIZE}\n")
            f.write(f"Train Samples: {len(train_dataset)}\n")
            f.write(f"Val Samples: {len(val_dataset)}\n\n")
            
            f.write("MODEL CONFIGURATION\n")
            f.write("-" * 70 + "\n")
            f.write(f"Model: SimpleDetector\n")
            f.write(f"Number of Classes: 10\n")
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
        
        np.savez(f"training_data_{training_start.strftime('%Y%m%d_%H%M%S')}.npz",
                 train_losses=np.array(training_stats['train_losses']),
                 val_losses=np.array(training_stats['val_losses']),
                 epoch_times=np.array(training_stats['epoch_times']),
                 learning_rates=np.array(training_stats['learning_rates']))
        print(f"Training data saved to: training_data_{training_start.strftime('%Y%m%d_%H%M%S')}.npz\n")
