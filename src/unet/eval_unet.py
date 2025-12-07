"""
Evaluation script for UNet segmentation model.
Computes IoU, Dice, and other segmentation metrics for 3-class segmentation.

Classes:
    0 = background (black)
    1 = spacecraft body (red)
    2 = solar panels (blue)
"""

import torch
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import argparse

from unet_model import UNetSegmentor
from dataset_unet import (
    SparkSegmentationDataset, 
    collate_fn_segmentation,
    CLASS_NAMES,
    NUM_CLASSES,
    CLASS_COLORS
)


def compute_confusion_matrix(pred, target, num_classes=NUM_CLASSES):
    """
    Compute confusion matrix for multi-class segmentation (vectorized, fast).
    
    Args:
        pred: Predicted class indices [H, W] or [B, H, W]
        target: Ground truth class indices [H, W] or [B, H, W]
        num_classes: Number of classes
    
    Returns:
        Confusion matrix of shape [num_classes, num_classes]
    """
    pred = pred.view(-1).long()
    target = target.view(-1).long()
    
    # Vectorized: use target * num_classes + pred as linear index
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
    
    # Per-class metrics
    per_class = {}
    for c in range(num_classes):
        tp = cm[c, c].item()
        fp = cm[:, c].sum().item() - tp
        fn = cm[c, :].sum().item() - tp
        tn = cm.sum().item() - tp - fp - fn
        
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
            'support': (cm[c, :].sum().item())  # Number of true samples
        }
    
    # Overall accuracy
    accuracy = cm.diag().sum().item() / cm.sum().item() if cm.sum().item() > 0 else 0.0
    
    # Mean metrics (excluding background optionally)
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
    """
    Convert class index mask to RGB visualization.
    
    Args:
        mask: numpy array [H, W] with class indices
    
    Returns:
        RGB image [H, W, 3]
    """
    H, W = mask.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    
    for class_idx, color in CLASS_COLORS.items():
        rgb[mask == class_idx] = color
    
    return rgb


def evaluate_model(model, dataloader, device, save_examples=False, save_path=None, use_fp16=True):
    """
    Evaluate UNet model on a dataset (optimized).
    
    Args:
        model: UNet model
        dataloader: DataLoader with evaluation data
        device: Device to run evaluation on
        save_examples: Whether to save example predictions
        save_path: Path to save results
        use_fp16: Use half precision for faster inference
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    # Accumulate confusion matrix across all batches
    total_cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    
    # Per-satellite-class confusion matrices
    satellite_cms = {}
    
    example_count = 0
    max_examples = 20
    
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_fp16):
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            images = batch["images"].to(device, non_blocking=True)
            masks = batch["masks"].to(device, non_blocking=True)
            satellite_classes = batch["satellite_classes"]
            
            outputs = model(images)
            pred_masks = outputs['pred_masks']  # [B, H, W] class indices
            
            # Accumulate confusion matrix for entire batch at once (vectorized)
            batch_cm = compute_confusion_matrix(pred_masks, masks)
            total_cm += batch_cm.to(device)
            
            # Per satellite class (still need loop but less frequent)
            unique_sats = set(satellite_classes)
            for sat_cls in unique_sats:
                mask_indices = [i for i, s in enumerate(satellite_classes) if s == sat_cls]
                if sat_cls not in satellite_cms:
                    satellite_cms[sat_cls] = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
                sat_cm = compute_confusion_matrix(
                    pred_masks[mask_indices], 
                    masks[mask_indices]
                )
                satellite_cms[sat_cls] += sat_cm.to(device)
            
            # Save example predictions
            if save_examples and save_path and example_count < max_examples:
                os.makedirs(os.path.join(save_path, "examples"), exist_ok=True)
                
                for i in range(min(images.size(0), max_examples - example_count)):
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    # Denormalize image
                    img = images[i].cpu()
                    img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    img = img + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    img = img.clamp(0, 1)
                    
                    axes[0].imshow(img.permute(1, 2, 0).numpy())
                    axes[0].set_title("Input Image")
                    axes[0].axis('off')
                    
                    # Ground truth as RGB
                    gt_rgb = class_mask_to_rgb(masks[i].cpu().numpy())
                    axes[1].imshow(gt_rgb)
                    axes[1].set_title("Ground Truth")
                    axes[1].axis('off')
                    
                    # Prediction as RGB
                    pred_rgb = class_mask_to_rgb(pred_masks[i].cpu().numpy())
                    axes[2].imshow(pred_rgb)
                    axes[2].set_title("Prediction")
                    axes[2].axis('off')
                    
                    plt.suptitle(f"Satellite: {satellite_classes[i]}", fontsize=12)
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(save_path, "examples", f"example_{example_count + i}.png"), 
                        dpi=150, bbox_inches='tight'
                    )
                    plt.close()
                
                example_count += images.size(0)
    
    # Move confusion matrices to CPU
    total_cm = total_cm.cpu()
    satellite_cms = {k: v.cpu() for k, v in satellite_cms.items()}
    
    # Compute overall metrics
    overall_metrics = compute_metrics_from_cm(total_cm)
    
    # Compute per-satellite-class metrics
    satellite_metrics = {}
    for sat_cls, cm in satellite_cms.items():
        satellite_metrics[sat_cls] = compute_metrics_from_cm(cm)
    
    return {
        'overall': overall_metrics,
        'per_satellite': satellite_metrics,
        'confusion_matrix': total_cm.numpy()
    }


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = UNetSegmentor.load_pretrained(
        args.model_path, 
        device=device,
        use_tiny_fallback=args.use_tiny,
        use_small_fallback=args.use_small
    )
    model = model.to(device)
    print("Model loaded successfully!")
    print(f"  - Input channels: {model.n_channels}")
    print(f"  - Output classes: {model.n_classes}")
    print(f"  - Model variant: {'UNetTiny' if model.use_tiny else ('UNetSmall' if model.use_small else 'UNet')}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = SparkSegmentationDataset(
        csv_path=args.val_csv,
        image_root=args.image_root,
        mask_root=args.mask_root,
        split="val",
        target_size=(args.target_size, args.target_size),
        augment=False
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=7,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=collate_fn_segmentation
    )
    
    print(f"Evaluating on {len(dataset)} samples...")
    print(f"Batch size: {args.batch_size}, Workers: 7")
    
    # Optional: torch.compile for PyTorch 2.x speedup
    if hasattr(torch, 'compile') and args.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode='reduce-overhead')
    
    # Evaluate
    results = evaluate_model(
        model, 
        dataloader, 
        device,
        use_fp16=args.fp16,
        save_examples=args.save_examples,
        save_path=args.output_path
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    print("\n📊 Overall Metrics:")
    print(f"  Pixel Accuracy:     {results['overall']['accuracy']:.4f}")
    print(f"  Mean IoU:           {results['overall']['mean_iou']:.4f}")
    print(f"  Mean IoU (no bg):   {results['overall']['mean_iou_no_bg']:.4f}")
    print(f"  Mean Dice:          {results['overall']['mean_dice']:.4f}")
    print(f"  Mean Dice (no bg):  {results['overall']['mean_dice_no_bg']:.4f}")
    
    print("\n📋 Per-Class Metrics:")
    for c in range(NUM_CLASSES):
        m = results['overall']['per_class'][c]
        print(f"\n  {CLASS_NAMES[c]}:")
        print(f"    IoU:       {m['iou']:.4f}")
        print(f"    Dice:      {m['dice']:.4f}")
        print(f"    Precision: {m['precision']:.4f}")
        print(f"    Recall:    {m['recall']:.4f}")
        print(f"    Support:   {m['support']:,}")
    
    # Save results
    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)
        
        # Save metrics as JSON
        metrics_path = os.path.join(args.output_path, "metrics.json")
        json_results = {
            'overall': {
                'accuracy': results['overall']['accuracy'],
                'mean_iou': results['overall']['mean_iou'],
                'mean_iou_no_bg': results['overall']['mean_iou_no_bg'],
                'mean_dice': results['overall']['mean_dice'],
                'mean_dice_no_bg': results['overall']['mean_dice_no_bg'],
                'per_class': {CLASS_NAMES[k]: v for k, v in results['overall']['per_class'].items()}
            },
            'per_satellite': {k: {
                'mean_iou': v['mean_iou'],
                'mean_dice': v['mean_dice']
            } for k, v in results['per_satellite'].items()}
        }
        with open(metrics_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"\n✅ Metrics saved to {metrics_path}")
        
        # Save confusion matrix plot
        cm_path = os.path.join(args.output_path, "confusion_matrix.png")
        plot_confusion_matrix(results['confusion_matrix'], CLASS_NAMES, cm_path)
        print(f"✅ Confusion matrix saved to {cm_path}")
        
        # Save detailed report
        report_path = os.path.join(args.output_path, "evaluation_report.txt")
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("UNET 3-CLASS SEGMENTATION EVALUATION REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("CLASSES:\n")
            for c in range(NUM_CLASSES):
                f.write(f"  {c}: {CLASS_NAMES[c]}\n")
            
            f.write("\n\nOVERALL METRICS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Pixel Accuracy:     {results['overall']['accuracy']:.6f}\n")
            f.write(f"Mean IoU:           {results['overall']['mean_iou']:.6f}\n")
            f.write(f"Mean IoU (no bg):   {results['overall']['mean_iou_no_bg']:.6f}\n")
            f.write(f"Mean Dice:          {results['overall']['mean_dice']:.6f}\n")
            f.write(f"Mean Dice (no bg):  {results['overall']['mean_dice_no_bg']:.6f}\n")
            
            f.write("\n\nPER-CLASS METRICS\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Class':<20} {'IoU':>10} {'Dice':>10} {'Precision':>10} {'Recall':>10} {'Support':>12}\n")
            f.write("-" * 70 + "\n")
            for c in range(NUM_CLASSES):
                m = results['overall']['per_class'][c]
                f.write(f"{CLASS_NAMES[c]:<20} {m['iou']:>10.4f} {m['dice']:>10.4f} "
                       f"{m['precision']:>10.4f} {m['recall']:>10.4f} {m['support']:>12,}\n")
            
            f.write("\n\nPER-SATELLITE METRICS\n")
            f.write("-" * 70 + "\n")
            for sat_cls, m in results['per_satellite'].items():
                f.write(f"\n{sat_cls}:\n")
                f.write(f"  Mean IoU:  {m['mean_iou']:.4f}\n")
                f.write(f"  Mean Dice: {m['mean_dice']:.4f}\n")
            
            f.write("\n\nCONFUSION MATRIX\n")
            f.write("-" * 70 + "\n")
            f.write("Rows: True labels, Columns: Predicted labels\n\n")
            f.write(f"{'':>15}")
            for name in CLASS_NAMES:
                f.write(f"{name:>15}")
            f.write("\n")
            for i, name in enumerate(CLASS_NAMES):
                f.write(f"{name:>15}")
                for j in range(NUM_CLASSES):
                    f.write(f"{results['confusion_matrix'][i, j]:>15,}")
                f.write("\n")
            
            f.write("\n" + "=" * 70 + "\n")
        print(f"✅ Report saved to {report_path}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate UNet 3-class segmentation model")
    
    parser.add_argument("--model_path", type=str, default="model_weights_unet/unet_best",
                        help="Path to saved model")
    parser.add_argument("--val_csv", type=str, default="/project/scratch/p200981/spark2024/val.csv",
                        help="Path to validation CSV")
    parser.add_argument("--image_root", type=str, default="/project/scratch/p200981/spark2024/images",
                        help="Path to images root")
    parser.add_argument("--mask_root", type=str, default="/project/scratch/p200981/spark2024/mask",
                        help="Path to masks root")
    parser.add_argument("--target_size", type=int, default=512,
                        help="Target image size")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--output_path", type=str, default="evaluation_results_unet",
                        help="Path to save evaluation results")
    parser.add_argument("--save_examples", action="store_true",
                        help="Save example predictions")
    parser.add_argument("--use_tiny", action="store_true", default=True,
                        help="Use UNetTiny model (for old checkpoints without variant info)")
    parser.add_argument("--use_small", action="store_true",
                        help="Use UNetSmall model")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Use FP16 mixed precision for faster inference")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile for faster inference (PyTorch 2.x)")
    
    args = parser.parse_args()
    main(args)
