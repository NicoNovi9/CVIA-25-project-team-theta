"""
Distributed training script for YOLO model on SPARK dataset.

This script handles:
- Loading configuration from YAML
- Converting SPARK dataset to YOLO format
- Training with DDP support via Ultralytics
- Saving checkpoints and training reports

Usage:
    # Single GPU:
    python train_yolo.py --config config_yolo.yaml
    
    # Multi-GPU (DDP via Ultralytics - RECOMMENDED):
    python train_yolo.py --config config_yolo.yaml --device 0,1,2,3
    
    # SLURM (single node, 4 GPUs):
    sbatch scripts/yolo_train.sh
"""

import os
import sys
import argparse
import datetime
import time
import yaml
import numpy as np
from pathlib import Path

import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset_yolo import (
    convert_spark_to_yolo,
    get_augmentation_params,
    verify_yolo_dataset,
    CLASS_NAMES
)
from yolo_model import YOLODetector, get_model_info


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train YOLO on SPARK dataset")
    parser.add_argument("--config", type=str, default="src/yolo/config_yolo.yaml",
                        help="Path to configuration file")
    parser.add_argument("--device", type=str, default=None,
                        help="Device(s) to use (e.g., '0', '0,1,2,3')")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--force_convert", action="store_true",
                        help="Force dataset conversion even if already exists")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    print("=" * 70)
    print("YOLO TRAINING - SPARK DETECTION DATASET")
    print("=" * 70)
    print(f"Configuration: {args.config}")
    print("=" * 70 + "\n")
    
    # ==========================================================================
    # DATASET PREPARATION
    # ==========================================================================
    data_config = config['data']
    data_root = data_config['data_root']
    yolo_dataset_path = data_config['yolo_dataset_path']
    
    print("[1/4] Preparing YOLO dataset...")
    
    # Convert SPARK to YOLO format
    data_yaml_path = convert_spark_to_yolo(
        data_root=data_root,
        output_path=yolo_dataset_path,
        train_csv=data_config['train_csv'],
        val_csv=data_config['val_csv'],
        image_root=data_config['image_root'],
        num_workers=8,
        force_recreate=args.force_convert
    )
    
    # Verify dataset
    verify_result = verify_yolo_dataset(yolo_dataset_path)
    if not verify_result['valid']:
        print(f"[ERROR] Dataset verification failed: {verify_result['errors']}")
        sys.exit(1)
    
    print(f"[Dataset] Train: {verify_result['stats'].get('train', {}).get('images', 0)} images")
    print(f"[Dataset] Val: {verify_result['stats'].get('val', {}).get('images', 0)} images")
    print()
    
    # ==========================================================================
    # MODEL CONFIGURATION
    # ==========================================================================
    model_config = config['model']
    training_config = config['training']
    aug_config = config['augmentation']
    output_config = config['output']
    
    print("[2/4] Configuring model...")
    
    model_size = model_config['size']
    model_info = get_model_info(model_size)
    
    print(f"Model: {model_info['name']}")
    print(f"  Parameters: {model_info['params']}")
    print(f"  Speed: {model_info['speed']}")
    print(f"  Accuracy: {model_info['accuracy']}")
    print()
    
    # Create detector
    detector = YOLODetector(
        model_size=model_config['size'],
        model_base=model_config['base'],
        num_classes=len(CLASS_NAMES),
        class_names=CLASS_NAMES,
        pretrained_backbone=model_config['pretrained_backbone']
    )
    
    # ==========================================================================
    # AUGMENTATION CONFIGURATION
    # ==========================================================================
    aug_level = aug_config['level']
    
    if aug_level == 'custom':
        # Use parameters from config directly
        augmentation_params = {
            'fliplr': aug_config.get('fliplr', 0.5),
            'flipud': aug_config.get('flipud', 0.1),
            'degrees': aug_config.get('degrees', 0.0),
            'translate': aug_config.get('translate', 0.1),
            'scale': aug_config.get('scale', 0.1),
            'shear': aug_config.get('shear', 0.0),
            'perspective': aug_config.get('perspective', 0.0),
            'hsv_h': aug_config.get('hsv_h', 0.015),
            'hsv_s': aug_config.get('hsv_s', 0.7),
            'hsv_v': aug_config.get('hsv_v', 0.4),
            'mosaic': aug_config.get('mosaic', 1.0),
            'mixup': aug_config.get('mixup', 0.0),
            'copy_paste': aug_config.get('copy_paste', 0.0),
            'auto_augment': aug_config.get('auto_augment', ''),
        }
    else:
        # Use preset level
        augmentation_params = get_augmentation_params(aug_level)
    
    print(f"[3/4] Augmentation level: {aug_level}")
    print(f"  Flip LR: {augmentation_params['fliplr']}")
    print(f"  Flip UD: {augmentation_params['flipud']}")
    print(f"  Degrees: {augmentation_params['degrees']}")
    print(f"  Mosaic: {augmentation_params['mosaic']}")
    print()
    
    # ==========================================================================
    # DEVICE CONFIGURATION
    # ==========================================================================
    # Check if we're in a SLURM distributed environment
    is_slurm_distributed = (
        'SLURM_NTASKS' in os.environ and 
        int(os.environ.get('SLURM_NTASKS', 1)) > 1
    )
    
    # Determine device - Ultralytics handles DDP internally when multiple GPUs specified
    if args.device:
        device = args.device
    elif is_slurm_distributed:
        # In SLURM multi-task environment, use single GPU per task
        # SLURM handles distribution, not Ultralytics
        local_rank = int(os.environ.get('SLURM_LOCALID', 0))
        device = local_rank
        print(f"[SLURM Distributed] Task {os.environ.get('SLURM_PROCID', 0)}, Local GPU: {local_rank}")
    elif torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            # Use all available GPUs - Ultralytics will spawn DDP workers
            device = ','.join(str(i) for i in range(n_gpus))
        else:
            device = 0
    else:
        device = 'cpu'
    
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA devices available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print()
    
    # ==========================================================================
    # TRAINING
    # ==========================================================================
    print("[4/4] Starting training...")
    print("-" * 70)
    print(f"Epochs: {training_config['epochs']}")
    print(f"Image size: {training_config['imgsz']}")
    print(f"Batch size: {training_config['batch_size']} (per GPU)")
    print(f"Patience: {training_config['patience']}")
    print(f"Learning rate: {training_config['lr0']}")
    print(f"AMP: {training_config['amp']}")
    print("-" * 70 + "\n")
    
    training_start = datetime.datetime.now()
    
    # Resume from checkpoint if specified
    resume = args.resume or training_config.get('resume', False)
    
    # Adjust save path for distributed training to avoid conflicts
    save_name = output_config['name'] or f"{model_config['base']}{model_config['size']}_{training_config['imgsz']}"
    if is_slurm_distributed:
        # Each SLURM task gets unique save directory
        rank = int(os.environ.get('SLURM_PROCID', 0))
        save_name = f"{save_name}_rank{rank}"
    
    # Run training
    try:
        results = detector.train(
            data_yaml=data_yaml_path,
            epochs=training_config['epochs'],
            imgsz=training_config['imgsz'],
            batch=training_config['batch_size'],
            patience=training_config['patience'],
            workers=training_config['workers'],
            device=device,
            project=output_config['save_path'],
            name=save_name,
            amp=training_config['amp'],
            resume=resume,
            save_period=output_config['save_period'],
            verbose=True,
            **augmentation_params
        )
        
        training_end = datetime.datetime.now()
        training_duration = (training_end - training_start).total_seconds()
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"Duration: {training_duration/3600:.2f} hours ({training_duration/60:.1f} minutes)")
        print(f"Results saved to: {output_config['save_path']}")
        print("=" * 70 + "\n")
        
        # ==========================================================================
        # SAVE TRAINING REPORT
        # ==========================================================================
        report_dir = Path("training_reports")
        report_dir.mkdir(exist_ok=True)
        
        report_file = report_dir / f"training_report_yolo_{training_start.strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("TRAINING REPORT - YOLO Detector Model\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("GENERAL INFORMATION\n")
            f.write("-" * 70 + "\n")
            f.write(f"Training Start: {training_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Training End:   {training_end.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Duration: {training_duration/3600:.2f} hours\n")
            f.write(f"Configuration: {args.config}\n")
            f.write("\n")
            
            f.write("MODEL CONFIGURATION\n")
            f.write("-" * 70 + "\n")
            f.write(f"Model: {model_config['base']}{model_config['size']}\n")
            f.write(f"Number of Classes: {len(CLASS_NAMES)}\n")
            f.write(f"Classes: {CLASS_NAMES}\n")
            f.write(f"Pretrained Backbone: {model_config['pretrained_backbone']}\n")
            f.write("\n")
            
            f.write("TRAINING PARAMETERS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Epochs: {training_config['epochs']}\n")
            f.write(f"Image Size: {training_config['imgsz']}\n")
            f.write(f"Batch Size: {training_config['batch_size']}\n")
            f.write(f"Patience: {training_config['patience']}\n")
            f.write(f"Learning Rate: {training_config['lr0']}\n")
            f.write(f"AMP: {training_config['amp']}\n")
            f.write(f"Device: {device}\n")
            f.write("\n")
            
            f.write("AUGMENTATION\n")
            f.write("-" * 70 + "\n")
            f.write(f"Level: {aug_level}\n")
            for key, value in augmentation_params.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            f.write("=" * 70 + "\n")
        
        print(f"Training report saved to: {report_file}")
            
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
