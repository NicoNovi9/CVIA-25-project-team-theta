"""
Evaluation script for SegFormer segmentation model.
Computes IoU, Dice, and other segmentation metrics for 3-class segmentation.

Classes:
    0 = background (black)
    1 = spacecraft body (red)
    2 = solar panels (blue)

Usage:
    python eval_segformer.py \
        --model_path model_weights_segformer/segformer_best \
        --config config_segformer.yaml
"""

import torch
from torch.utils.data import DataLoader
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import argparse
import yaml

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'unet'))
from dataset_unet import (
    SparkSegmentationDataset, 
    collate_fn_segmentation,
    CLASS_NAMES,
    NUM_CLASSES,
    CLASS_COLORS
)

from segformer_model import SegFormerSegmentor


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def compute_confusion_matrix(pred, target, num_classes=NUM_CLASSES):
    """
    Compute confusion matrix for multi-class segmentation (vectorized).
    
    Args:
        pred: Predicted class indices [H, W] or [B, H, W]
        target: Ground truth class indices [H, W] or [B, H, W]
        num_classes: Number of classes
    
    Returns:
        Confusion matrix of shape [num_classes, num_classes]
    """
    pred = pred.view(-1).long()
    target = target.view(-1).long()
    
    indices = target * num_classes + pred
    cm = torch.bincount(indices, minlength=num_classes * num_classes)
    cm = cm.reshape(num_classes, num_classes)
    
    return cm


def compute_metrics_from_cm(cm):
    """
    Compute per-class and mean metrics from confusion matrix.
    
    Args:
        cm: Confusion matrix [num_classes, num_classes]
    
    Returns:
        Dictionary with metrics
    """
    num_classes = cm.shape[0]
    
    per_class = {}
    for c in range(num_classes):
        tp = cm[c, c].item()
        fp = cm[:, c].sum().item() - tp
        fn = cm[c, :].sum().item() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class[c] = {
            'precision': precision,
            'recall': recall,
            'iou': iou,
            'dice': dice,
            'f1': f1,
            'support': cm[c, :].sum().item()
        }
    
    accuracy = cm.diag().sum().item() / cm.sum().item() if cm.sum().item() > 0 else 0.0
    
    mean_iou = np.mean([per_class[c]['iou'] for c in range(num_classes)])
    mean_iou_no_bg = np.mean([per_class[c]['iou'] for c in range(1, num_classes)])
    mean_dice = np.mean([per_class[c]['dice'] for c in range(num_classes)])
    mean_dice_no_bg = np.mean([per_class[c]['dice'] for c in range(1, num_classes)])
    
    return {
        'per_class': per_class,
        'accuracy': accuracy,
        'mean_iou': mean_iou,
        'mean_iou_no_bg': mean_iou_no_bg,
        'mean_dice': mean_dice,
        'mean_dice_no_bg': mean_dice_no_bg
    }


def class_mask_to_rgb(mask):
    """Convert class index mask to RGB visualization."""
    H, W = mask.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    
    for class_idx, color in CLASS_COLORS.items():
        rgb[mask == class_idx] = color
    
    return rgb


def evaluate_model(model, dataloader, device, save_examples=False, save_path=None, use_fp16=True):
    """
    Evaluate SegFormer model on a dataset.
    
    Args:
        model: SegFormer model
        dataloader: DataLoader with evaluation data
        device: Device to run evaluation on
        save_examples: Whether to save example predictions
        save_path: Path to save results
        use_fp16: Use half precision for faster inference
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    total_cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    satellite_cms = {}
    
    example_count = 0
    max_examples = 20
    
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_fp16):
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            images = batch["images"].to(device, non_blocking=True)
            masks = batch["masks"].to(device, non_blocking=True)
            satellite_classes = batch["satellite_classes"]
            
            outputs = model(images)
            pred_masks = outputs['pred_masks']
            
            # Accumulate confusion matrix
            batch_cm = compute_confusion_matrix(pred_masks, masks)
            total_cm += batch_cm.to(device)
            
            # Per satellite class
            unique_sats = set(satellite_classes)
            for sat_cls in unique_sats:
                mask_indices = [i for i, s in enumerate(satellite_classes) if s == sat_cls]
                if sat_cls not in satellite_cms:
                    satellite_cms[sat_cls] = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
                sat_cm = compute_confusion_matrix(pred_masks[mask_indices], masks[mask_indices])
                satellite_cms[sat_cls] += sat_cm.to(device)
            
            # Save examples
            if save_examples and save_path and example_count < max_examples:
                os.makedirs(os.path.join(save_path, "examples"), exist_ok=True)
                
                for i in range(min(images.size(0), max_examples - example_count)):
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    # Denormalize image
                    img = images[i].cpu().float()
                    img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    img = img + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    img = img.clamp(0, 1)
                    
                    axes[0].imshow(img.permute(1, 2, 0).numpy())
                    axes[0].set_title("Input Image")
                    axes[0].axis('off')
                    
                    gt_rgb = class_mask_to_rgb(masks[i].cpu().numpy())
                    axes[1].imshow(gt_rgb)
                    axes[1].set_title("Ground Truth")
                    axes[1].axis('off')
                    
                    pred_rgb = class_mask_to_rgb(pred_masks[i].cpu().numpy())
                    axes[2].imshow(pred_rgb)
                    axes[2].set_title("Prediction")
                    axes[2].axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_path, "examples", f"example_{example_count}.png"), dpi=150)
                    plt.close()
                    example_count += 1
    
    # Compute metrics
    overall_metrics = compute_metrics_from_cm(total_cm.cpu())
    
    # Per-satellite metrics
    satellite_metrics = {}
    for sat_cls, cm in satellite_cms.items():
        satellite_metrics[sat_cls] = compute_metrics_from_cm(cm.cpu())
    
    results = {
        'overall': overall_metrics,
        'per_satellite': satellite_metrics,
        'confusion_matrix': total_cm.cpu().numpy().tolist()
    }
    
    return results


def print_evaluation_results(results):
    """Print evaluation results in a formatted table."""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    overall = results['overall']
    
    print("\nOVERALL METRICS:")
    print("-" * 50)
    print(f"  Pixel Accuracy:     {overall['accuracy']*100:.2f}%")
    print(f"  Mean IoU (all):     {overall['mean_iou']*100:.2f}%")
    print(f"  Mean IoU (no bg):   {overall['mean_iou_no_bg']*100:.2f}%")
    print(f"  Mean Dice (all):    {overall['mean_dice']*100:.2f}%")
    print(f"  Mean Dice (no bg):  {overall['mean_dice_no_bg']*100:.2f}%")
    
    print("\nPER-CLASS METRICS:")
    print("-" * 70)
    print(f"{'Class':<20} {'IoU':>10} {'Dice':>10} {'Precision':>10} {'Recall':>10}")
    print("-" * 70)
    
    for c, metrics in overall['per_class'].items():
        class_name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"Class {c}"
        print(f"{class_name:<20} {metrics['iou']*100:>9.2f}% {metrics['dice']*100:>9.2f}% "
              f"{metrics['precision']*100:>9.2f}% {metrics['recall']*100:>9.2f}%")
    
    print("\nPER-SATELLITE METRICS (IoU):")
    print("-" * 70)
    
    for sat_cls, sat_metrics in sorted(results['per_satellite'].items()):
        mean_iou = sat_metrics['mean_iou_no_bg']
        print(f"  {sat_cls:<20}: mIoU (no bg) = {mean_iou*100:.2f}%")
    
    print("=" * 70)


def save_results(results, save_path):
    """Save evaluation results to JSON file."""
    os.makedirs(save_path, exist_ok=True)
    
    results_path = os.path.join(save_path, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Results saved to {results_path}")


def plot_confusion_matrix(cm, save_path):
    """Plot and save confusion matrix."""
    cm_np = np.array(cm)
    cm_normalized = cm_np.astype('float') / cm_np.sum(axis=1, keepdims=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw counts
    im1 = axes[0].imshow(cm_np, cmap='Blues')
    axes[0].set_title('Confusion Matrix (Counts)', fontweight='bold')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_xticks(range(NUM_CLASSES))
    axes[0].set_yticks(range(NUM_CLASSES))
    axes[0].set_xticklabels(CLASS_NAMES, rotation=45)
    axes[0].set_yticklabels(CLASS_NAMES)
    
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            axes[0].text(j, i, f'{cm_np[i, j]:,}', ha='center', va='center', fontsize=9)
    
    plt.colorbar(im1, ax=axes[0])
    
    # Normalized
    im2 = axes[1].imshow(cm_normalized, cmap='Blues', vmin=0, vmax=1)
    axes[1].set_title('Confusion Matrix (Normalized)', fontweight='bold')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_xticks(range(NUM_CLASSES))
    axes[1].set_yticks(range(NUM_CLASSES))
    axes[1].set_xticklabels(CLASS_NAMES, rotation=45)
    axes[1].set_yticklabels(CLASS_NAMES)
    
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            axes[1].text(j, i, f'{cm_normalized[i, j]:.2f}', ha='center', va='center', fontsize=9)
    
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "confusion_matrix.png"), dpi=150)
    plt.close()
    
    print(f"📊 Confusion matrix saved to {save_path}/confusion_matrix.png")


def parse_args():
    parser = argparse.ArgumentParser(description="SegFormer Evaluation")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to saved model directory")
    parser.add_argument("--config", type=str, default="src/segformer/config_segformer.yaml",
                        help="Path to config file")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save evaluation results (default: model_path)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of data loading workers")
    parser.add_argument("--save_examples", action="store_true",
                        help="Save example predictions")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"],
                        help="Dataset split to evaluate")
    parser.add_argument("--no_fp16", action="store_true",
                        help="Disable FP16 inference")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    data_cfg = config['data']
    training_cfg = config['training']
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Output directory
    output_dir = args.output_dir or os.path.join(args.model_path, "evaluation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print(f"\nLoading model from {args.model_path}...")
    model = SegFormerSegmentor.load_pretrained(args.model_path, device=device)
    model.to(device)
    model.eval()
    
    # Prepare dataset
    print(f"\nPreparing {args.split} dataset...")
    
    data_root = data_cfg['data_root']
    target_size = (training_cfg['image_size'], training_cfg['image_size'])
    
    csv_file = data_cfg['val_csv'] if args.split == 'val' else data_cfg['train_csv']
    
    dataset = SparkSegmentationDataset(
        csv_path=os.path.join(data_root, csv_file),
        image_root=os.path.join(data_root, data_cfg['image_root']),
        mask_root=os.path.join(data_root, data_cfg['mask_root']),
        split=args.split,
        target_size=target_size,
        augment=False
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn_segmentation
    )
    
    print(f"Evaluating on {len(dataset)} samples...")
    
    # Evaluate
    results = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        save_examples=args.save_examples,
        save_path=output_dir,
        use_fp16=not args.no_fp16
    )
    
    # Print and save results
    print_evaluation_results(results)
    save_results(results, output_dir)
    plot_confusion_matrix(results['confusion_matrix'], output_dir)
    
    print(f"\n✅ Evaluation complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
