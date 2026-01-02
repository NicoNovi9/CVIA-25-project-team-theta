"""
Evaluation script for YOLO model on SPARK dataset.

Computes:
- mAP (COCO metrics: mAP@0.5, mAP@0.5:0.95)
- Per-class AP
- Confusion matrix
- Classification metrics

Usage:
    python eval_yolo.py --model_path model_weights_yolo/yolo11n_640/weights/best.pt \
                        --data_yaml /project/scratch/p200981/spark2024_yolo/data.yaml
    
    # With SLURM (multi-GPU):
    srun python eval_yolo.py --model_path ... --data_yaml ...
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset_yolo import CLASS_NAMES, ID_TO_CLASS, CLASS_TO_ID

import warnings
warnings.filterwarnings("ignore")


def setup_distributed():
    """Initialize distributed environment."""
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ.get("SLURM_LOCALID", 0))
        
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)
        
        if "MASTER_ADDR" not in os.environ:
            import subprocess
            result = subprocess.run(
                ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]],
                capture_output=True, text=True
            )
            master = result.stdout.strip().split('\n')[0]
            os.environ["MASTER_ADDR"] = master
        
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29501"
        
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank, True
    
    elif "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank, True
    
    return 0, 1, 0, False


def cleanup_distributed():
    """Clean up distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLO model on SPARK dataset")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to YOLO model weights (.pt file)")
    parser.add_argument("--data_yaml", type=str, default=None,
                        help="Path to data.yaml for validation data")
    parser.add_argument("--data_root", type=str, 
                        default="/project/scratch/p200981/spark2024",
                        help="SPARK dataset root (if data_yaml not provided)")
    parser.add_argument("--output_dir", type=str, default="evaluation_results_yolo",
                        help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size for evaluation")
    parser.add_argument("--conf", type=float, default=0.001,
                        help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6,
                        help="IoU threshold for NMS")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use")
    
    return parser.parse_args()


def compute_classification_metrics(y_true, y_pred, class_names):
    """
    Compute classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
    
    Returns:
        Dictionary with metrics
    """
    # Overall accuracy
    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    
    # Classification report
    report = classification_report(
        y_true, y_pred, 
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    
    return {
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm
    }


def plot_confusion_matrix(cm, class_names, output_path, normalize=True):
    """Plot and save confusion matrix."""
    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
    else:
        cm_normalized = cm
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_normalized,
        display_labels=class_names
    )
    disp.plot(ax=ax, cmap='Blues', values_format='.2f' if normalize else 'd')
    
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''), fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_per_class_metrics(metrics_dict, class_names, output_path):
    """Plot per-class precision, recall, F1."""
    classes = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for name in class_names:
        if name in metrics_dict:
            classes.append(name)
            precisions.append(metrics_dict[name]['precision'])
            recalls.append(metrics_dict[name]['recall'])
            f1_scores.append(metrics_dict[name]['f1-score'])
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    bars1 = ax.bar(x - width, precisions, width, label='Precision', color='#2ecc71')
    bars2 = ax.bar(x, recalls, width, label='Recall', color='#3498db')
    bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', color='#e74c3c')
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Metrics', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    
    # Setup distributed
    rank, world_size, local_rank, is_distributed = setup_distributed()
    is_main = (rank == 0)
    
    if is_main:
        print("=" * 70)
        print("YOLO EVALUATION - SPARK DATASET")
        print("=" * 70)
        print(f"Model: {args.model_path}")
        print(f"Distributed: {is_distributed}")
        if is_distributed:
            print(f"World size: {world_size}")
        print("=" * 70 + "\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine device
    if args.device:
        device = args.device
    elif is_distributed:
        device = local_rank
    elif torch.cuda.is_available():
        device = 0
    else:
        device = 'cpu'
    
    if is_main:
        print(f"Device: {device}")
    
    # ==========================================================================
    # LOAD MODEL
    # ==========================================================================
    if is_main:
        print("\n[1/3] Loading model...")
    
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)
    
    model = YOLO(args.model_path)
    
    if is_main:
        print(f"✓ Model loaded: {args.model_path}")
    
    # ==========================================================================
    # RUN VALIDATION
    # ==========================================================================
    if is_main:
        print("\n[2/3] Running validation...")
    
    # If data_yaml not provided, try to find it
    data_yaml = args.data_yaml
    if not data_yaml:
        yolo_dataset_path = Path("/project/scratch/p200981/spark2024_yolo")
        data_yaml = yolo_dataset_path / "data.yaml"
        if not data_yaml.exists():
            print(f"[ERROR] data.yaml not found at {data_yaml}")
            print("Please provide --data_yaml or run train_yolo.py first to create the dataset")
            sys.exit(1)
        data_yaml = str(data_yaml)
    
    if is_main:
        print(f"Data: {data_yaml}")
    
    # Run YOLO validation
    results = model.val(
        data=data_yaml,
        imgsz=args.imgsz,
        batch=args.batch_size,
        conf=args.conf,
        iou=args.iou,
        device=device,
        verbose=True,
        plots=True,  # Generate plots
        save_json=True,  # Save COCO JSON results
    )
    
    if is_main:
        print("\n" + "-" * 70)
        print("VALIDATION RESULTS")
        print("-" * 70)
        
        # Extract metrics
        metrics = results.results_dict
        
        print(f"\nmAP@0.5:      {metrics.get('metrics/mAP50(B)', 0):.4f}")
        print(f"mAP@0.5:0.95: {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
        print(f"Precision:    {metrics.get('metrics/precision(B)', 0):.4f}")
        print(f"Recall:       {metrics.get('metrics/recall(B)', 0):.4f}")
        
        # Per-class AP
        if hasattr(results, 'ap_class_index') and hasattr(results, 'ap'):
            print("\nPer-Class mAP@0.5:")
            for i, cls_idx in enumerate(results.ap_class_index):
                class_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"class_{cls_idx}"
                ap = results.ap[i, 0]  # AP at IoU=0.5
                print(f"  {class_name}: {ap:.4f}")
        
        # ==========================================================================
        # RUN INFERENCE FOR CLASSIFICATION METRICS
        # ==========================================================================
        print("\n[3/3] Computing classification metrics...")
        
        # Load validation data info
        import yaml
        with open(data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
        
        val_images_dir = Path(data_config['path']) / data_config['val']
        val_labels_dir = Path(data_config['path']) / "labels" / "val"
        
        # Get predictions for all validation images
        y_true = []
        y_pred = []
        
        image_files = list(val_images_dir.glob("*.[jJpP][pPnN][gG]")) + \
                      list(val_images_dir.glob("*.jpeg"))
        
        print(f"Running inference on {len(image_files)} validation images...")
        
        # Run inference in batches
        batch_size = args.batch_size
        
        for i in tqdm(range(0, len(image_files), batch_size), desc="Inference"):
            batch_files = image_files[i:i+batch_size]
            
            # Run prediction
            preds = model.predict(
                source=[str(f) for f in batch_files],
                imgsz=args.imgsz,
                conf=0.001,  # Low threshold to get predictions
                device=device,
                verbose=False
            )
            
            for img_path, pred in zip(batch_files, preds):
                # Get ground truth from label file
                label_file = val_labels_dir / (img_path.stem + ".txt")
                
                if label_file.exists():
                    with open(label_file, 'r') as f:
                        line = f.readline().strip()
                        if line:
                            gt_class = int(line.split()[0])
                            y_true.append(gt_class)
                            
                            # Get best prediction
                            if len(pred.boxes) > 0:
                                # Get highest confidence prediction
                                best_idx = pred.boxes.conf.argmax()
                                pred_class = int(pred.boxes.cls[best_idx])
                                y_pred.append(pred_class)
                            else:
                                # No detection - use most common class as fallback
                                y_pred.append(0)
        
        # Compute classification metrics
        if len(y_true) > 0:
            cls_metrics = compute_classification_metrics(y_true, y_pred, CLASS_NAMES)
            
            print(f"\n" + "-" * 70)
            print("CLASSIFICATION METRICS")
            print("-" * 70)
            print(f"Overall Accuracy: {cls_metrics['accuracy']:.4f}")
            
            # Print per-class metrics
            print("\nPer-Class Metrics:")
            print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
            print("-" * 62)
            
            for class_name in CLASS_NAMES:
                if class_name in cls_metrics['report']:
                    m = cls_metrics['report'][class_name]
                    print(f"{class_name:<20} {m['precision']:>10.4f} {m['recall']:>10.4f} "
                          f"{m['f1-score']:>10.4f} {int(m['support']):>10}")
            
            # Macro/Weighted averages
            print("-" * 62)
            macro = cls_metrics['report']['macro avg']
            weighted = cls_metrics['report']['weighted avg']
            print(f"{'Macro Avg':<20} {macro['precision']:>10.4f} {macro['recall']:>10.4f} "
                  f"{macro['f1-score']:>10.4f}")
            print(f"{'Weighted Avg':<20} {weighted['precision']:>10.4f} {weighted['recall']:>10.4f} "
                  f"{weighted['f1-score']:>10.4f}")
            
            # Plot confusion matrix
            cm_path = output_dir / "confusion_matrix.png"
            plot_confusion_matrix(
                cls_metrics['confusion_matrix'],
                CLASS_NAMES,
                cm_path,
                normalize=True
            )
            print(f"\n✓ Confusion matrix saved to: {cm_path}")
            
            # Plot per-class metrics
            metrics_path = output_dir / "per_class_metrics.png"
            plot_per_class_metrics(cls_metrics['report'], CLASS_NAMES, metrics_path)
            print(f"✓ Per-class metrics plot saved to: {metrics_path}")
            
            # Save results to JSON
            results_json = {
                'model_path': args.model_path,
                'data_yaml': data_yaml,
                'mAP50': float(metrics.get('metrics/mAP50(B)', 0)),
                'mAP50_95': float(metrics.get('metrics/mAP50-95(B)', 0)),
                'precision': float(metrics.get('metrics/precision(B)', 0)),
                'recall': float(metrics.get('metrics/recall(B)', 0)),
                'classification_accuracy': float(cls_metrics['accuracy']),
                'per_class': {}
            }
            
            for class_name in CLASS_NAMES:
                if class_name in cls_metrics['report']:
                    m = cls_metrics['report'][class_name]
                    results_json['per_class'][class_name] = {
                        'precision': float(m['precision']),
                        'recall': float(m['recall']),
                        'f1': float(m['f1-score']),
                        'support': int(m['support'])
                    }
            
            json_path = output_dir / "evaluation_results.json"
            with open(json_path, 'w') as f:
                json.dump(results_json, f, indent=2)
            print(f"✓ Results saved to: {json_path}")
        
        print("\n" + "=" * 70)
        print("EVALUATION COMPLETED")
        print("=" * 70)
    
    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    main()
