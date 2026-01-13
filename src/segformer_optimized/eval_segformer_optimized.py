"""
Evaluation script for SegFormer Optimized segmentation model.
Computes IoU, Dice, and other metrics for YOLO-guided 3-class segmentation.

This evaluation:
1. Uses YOLO to detect bounding boxes
2. Extracts and prepares bbox region (pad/resize to 512)
3. Runs segmentation on bbox region
4. Maps prediction back to full image
5. Computes metrics on full-resolution predictions

Classes:
    0 = background (black)
    1 = spacecraft body (red)
    2 = solar panels (blue)

Usage:
    python eval_segformer_optimized.py \
        --model_path model_weights_segformer_optimized/segformer_optimized_best \
        --config config_segformer_optimized.yaml
"""

import torch
import torch.nn.functional as F
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
from PIL import Image

from dataset_segformer_optimized import (
    SparkSegmentationOptimizedDataset,
    collate_fn_segmentation_optimized,
    CLASS_NAMES,
    NUM_CLASSES,
    CLASS_COLORS
)
from segformer_optimized_model import SegFormerOptimizedSegmentor, BBoxProcessor


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def compute_confusion_matrix(pred, target, num_classes=NUM_CLASSES):
    """
    Compute confusion matrix for multi-class segmentation (vectorized).
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


def evaluate_model(model, dataloader, device, save_examples=False, save_path=None, use_fp16=True, 
                   dataset=None, data_root=None, image_root=None, mask_root=None, split='val'):
    """
    Evaluate SegFormer Optimized model on a dataset.
    
    Metrics are computed on the bbox crop predictions (512x512).
    For full-image metrics, see full evaluation mode.
    
    Args:
        model: The model to evaluate
        dataloader: Dataloader for the dataset
        device: Device to run evaluation on
        save_examples: Whether to save example visualizations
        save_path: Path to save examples
        use_fp16: Whether to use FP16 inference
        dataset: The dataset object (needed for full-image visualization)
        data_root: Root directory for data (needed for loading full images)
        image_root: Image directory path
        mask_root: Mask directory path
        split: Dataset split ('train' or 'val')
    """
    model.eval()
    
    total_cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    satellite_cms = {}
    
    # BBox statistics
    bbox_stats = {'padded': 0, 'downsampled': 0, 'total': 0}
    bbox_sizes = []
    
    example_count = 0
    max_examples = 20
    
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_fp16):
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            images = batch["images"].to(device, non_blocking=True)
            masks = batch["masks"].to(device, non_blocking=True)
            satellite_classes = batch["satellite_classes"]
            image_names = batch["image_names"]
            bboxes = batch["bboxes"]
            metadatas = batch["metadatas"]
            
            # Track bbox statistics
            for meta in metadatas:
                bbox_stats['total'] += 1
                if not meta['needs_downsampling']:
                    bbox_stats['padded'] += 1
                else:
                    bbox_stats['downsampled'] += 1
                bbox_sizes.append((meta['bbox_width'], meta['bbox_height']))
            
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
            
            # Save examples with full-image visualization
            if save_examples and save_path and example_count < max_examples and dataset is not None:
                os.makedirs(os.path.join(save_path, "examples"), exist_ok=True)
                
                for i in range(min(images.size(0), max_examples - example_count)):
                    # Get the dataset sample to retrieve full image and mask
                    sample_idx = batch_idx * dataloader.batch_size + i
                    if sample_idx >= len(dataset):
                        break
                    
                    row = dataset.df.iloc[sample_idx]
                    
                    # Load full image
                    img_path = os.path.join(image_root, row["Class"], split, row["Image name"])
                    full_img = Image.open(img_path)
                    if full_img.mode != "RGB":
                        full_img = full_img.convert("RGB")
                    full_img_np = np.array(full_img)
                    
                    # Load full mask
                    mask_path = os.path.join(mask_root, row["Class"], split, row["Mask name"])
                    full_mask = Image.open(mask_path)
                    if full_mask.mode != "RGB":
                        full_mask = full_mask.convert("RGB")
                    full_mask_np = np.array(full_mask)
                    
                    # Convert to class mask
                    from dataset_segformer_optimized import rgb_to_class_mask
                    full_gt_mask = rgb_to_class_mask(full_mask_np)
                    
                    # Get bbox from metadata
                    bbox = bboxes[i]
                    x_min, y_min, x_max, y_max = bbox
                    
                    # Create full prediction mask (initialize as background)
                    H, W, _ = full_img_np.shape
                    full_pred_mask = np.zeros((H, W), dtype=np.int64)
                    
                    # Map the prediction back to the bbox region
                    pred_crop = pred_masks[i].cpu().numpy()  # [512, 512]
                    meta = metadatas[i]
                    
                    # Extract the bbox region size
                    bbox_h = y_max - y_min
                    bbox_w = x_max - x_min
                    
                    # Resize prediction back to original bbox size
                    if not meta['needs_downsampling']:
                        # Was only padded - extract valid region
                        pred_resized = pred_crop[:bbox_h, :bbox_w]
                    else:
                        # Was padded to square then downsampled
                        # Step 1: Upsample from 512x512 back to square size
                        square_size = meta['square_size']
                        pred_tensor = torch.from_numpy(pred_crop).unsqueeze(0).unsqueeze(0).float()
                        pred_square = F.interpolate(
                            pred_tensor,
                            size=(square_size, square_size),
                            mode='nearest'
                        ).squeeze().numpy().astype(np.int64)
                        
                        # Step 2: Extract the original bbox region from the square
                        pred_resized = pred_square[:bbox_h, :bbox_w]
                    
                    # Place prediction in bbox region
                    full_pred_mask[y_min:y_max, x_min:x_max] = pred_resized
                    
                    # Create visualization
                    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                    
                    # Row 1: Full images
                    # Original image with bbox
                    axes[0, 0].imshow(full_img_np)
                    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                        fill=False, edgecolor='red', linewidth=2)
                    axes[0, 0].add_patch(rect)
                    axes[0, 0].set_title(f"Full Image with BBox\n{row['Image name']}\nClass: {row['Class']}")
                    axes[0, 0].axis('off')
                    
                    # Full ground truth mask
                    full_gt_rgb = class_mask_to_rgb(full_gt_mask)
                    axes[0, 1].imshow(full_gt_rgb)
                    axes[0, 1].set_title("Full Ground Truth Mask")
                    axes[0, 1].axis('off')
                    
                    # Full prediction mask
                    full_pred_rgb = class_mask_to_rgb(full_pred_mask)
                    axes[0, 2].imshow(full_pred_rgb)
                    axes[0, 2].set_title("Full Prediction Mask")
                    axes[0, 2].axis('off')
                    
                    # Row 2: Crops and details
                    # Denormalize cropped image
                    img_crop = images[i].cpu().float()
                    img_crop = img_crop * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    img_crop = img_crop + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    img_crop = img_crop.clamp(0, 1)
                    
                    bbox_info = f"BBox: {meta['bbox_width']}x{meta['bbox_height']}\n"
                    if not meta['needs_downsampling']:
                        bbox_info += "(Padded to 512x512)"
                    else:
                        bbox_info += f"(Padded to {meta['square_size']}x{meta['square_size']}, then downsampled)"
                    
                    axes[1, 0].imshow(img_crop.permute(1, 2, 0).numpy())
                    axes[1, 0].set_title(f"BBox Crop (512x512)\n{bbox_info}")
                    axes[1, 0].axis('off')
                    
                    # Crop GT mask
                    gt_crop_rgb = class_mask_to_rgb(masks[i].cpu().numpy())
                    axes[1, 1].imshow(gt_crop_rgb)
                    axes[1, 1].set_title("Crop GT Mask")
                    axes[1, 1].axis('off')
                    
                    # Crop prediction mask
                    pred_crop_rgb = class_mask_to_rgb(pred_masks[i].cpu().numpy())
                    axes[1, 2].imshow(pred_crop_rgb)
                    axes[1, 2].set_title("Crop Prediction Mask")
                    axes[1, 2].axis('off')
                    
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
    
    # Compute bbox size statistics
    if bbox_sizes:
        widths = [s[0] for s in bbox_sizes]
        heights = [s[1] for s in bbox_sizes]
        bbox_size_stats = {
            'mean_width': np.mean(widths),
            'mean_height': np.mean(heights),
            'median_width': np.median(widths),
            'median_height': np.median(heights),
            'min_width': np.min(widths),
            'min_height': np.min(heights),
            'max_width': np.max(widths),
            'max_height': np.max(heights),
        }
    else:
        bbox_size_stats = {}
    
    results = {
        'overall': overall_metrics,
        'per_satellite': satellite_metrics,
        'confusion_matrix': total_cm.cpu().numpy().tolist(),
        'bbox_stats': bbox_stats,
        'bbox_size_stats': bbox_size_stats
    }
    
    return results


def print_evaluation_results(results):
    """Print evaluation results in a formatted table."""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS (SegFormer Optimized)")
    print("=" * 70)
    
    overall = results['overall']
    bbox_stats = results.get('bbox_stats', {})
    bbox_size_stats = results.get('bbox_size_stats', {})
    
    print("\nBBOX STATISTICS:")
    print("-" * 50)
    total = bbox_stats.get('total', 0)
    if total > 0:
        padded = bbox_stats.get('padded', 0)
        downsampled = bbox_stats.get('downsampled', 0)
        print(f"  Total samples: {total}")
        print(f"  Padded (preserved resolution): {padded} ({100*padded/total:.1f}%)")
        print(f"  Downsampled: {downsampled} ({100*downsampled/total:.1f}%)")
        
        if bbox_size_stats:
            print(f"\n  BBox Sizes:")
            print(f"    Mean:   {bbox_size_stats['mean_width']:.0f} x {bbox_size_stats['mean_height']:.0f}")
            print(f"    Median: {bbox_size_stats['median_width']:.0f} x {bbox_size_stats['median_height']:.0f}")
            print(f"    Range:  [{bbox_size_stats['min_width']:.0f}-{bbox_size_stats['max_width']:.0f}] x "
                  f"[{bbox_size_stats['min_height']:.0f}-{bbox_size_stats['max_height']:.0f}]")
    
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
    
    print("\nPER-SATELLITE METRICS (mIoU no bg):")
    print("-" * 70)
    
    for sat_cls, sat_metrics in sorted(results['per_satellite'].items()):
        mean_iou = sat_metrics['mean_iou_no_bg']
        print(f"  {sat_cls:<20}: {mean_iou*100:.2f}%")
    
    print("=" * 70)


def save_results(results, save_path):
    """Save evaluation results to JSON file."""
    import numpy as np
    
    def convert_to_serializable(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    os.makedirs(save_path, exist_ok=True)
    
    # Convert all numpy types to native Python types
    results_serializable = convert_to_serializable(results)
    
    results_path = os.path.join(save_path, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\n📄 Results saved to {results_path}")


def plot_confusion_matrix(cm, save_path):
    """Plot and save confusion matrix."""
    cm_np = np.array(cm)
    cm_normalized = cm_np.astype('float') / cm_np.sum(axis=1, keepdims=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
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


def plot_bbox_distribution(bbox_sizes, save_path, target_size=512):
    """Plot distribution of bounding box sizes."""
    widths = [s[0] for s in bbox_sizes]
    heights = [s[1] for s in bbox_sizes]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Width distribution
    axes[0].hist(widths, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(x=target_size, color='r', linestyle='--', label=f'Target size ({target_size})')
    axes[0].set_xlabel('BBox Width')
    axes[0].set_ylabel('Count')
    axes[0].set_title('BBox Width Distribution')
    axes[0].legend()
    
    # Height distribution
    axes[1].hist(heights, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=target_size, color='r', linestyle='--', label=f'Target size ({target_size})')
    axes[1].set_xlabel('BBox Height')
    axes[1].set_ylabel('Count')
    axes[1].set_title('BBox Height Distribution')
    axes[1].legend()
    
    # Scatter plot
    axes[2].scatter(widths, heights, alpha=0.3, s=10)
    axes[2].axhline(y=target_size, color='r', linestyle='--', alpha=0.5)
    axes[2].axvline(x=target_size, color='r', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('Width')
    axes[2].set_ylabel('Height')
    axes[2].set_title('BBox Width vs Height')
    axes[2].set_aspect('equal')
    
    # Count in each quadrant
    small = sum(1 for w, h in bbox_sizes if w <= target_size and h <= target_size)
    large = len(bbox_sizes) - small
    axes[2].text(0.05, 0.95, f'Padded: {small}\nDownsampled: {large}', 
                 transform=axes[2].transAxes, verticalalignment='top',
                 fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "bbox_distribution.png"), dpi=150)
    plt.close()
    
    print(f"📊 BBox distribution saved to {save_path}/bbox_distribution.png")


def parse_args():
    parser = argparse.ArgumentParser(description="SegFormer Optimized Evaluation")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to saved model directory")
    parser.add_argument("--config", type=str, default="src/segformer_optimized/config_segformer_optimized.yaml",
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
    
    config = load_config(args.config)
    data_cfg = config['data']
    training_cfg = config['training']
    bbox_cfg = config.get('bbox', {})
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    output_dir = args.output_dir or os.path.join(args.model_path, "evaluation")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nLoading model from {args.model_path}...")
    model = SegFormerOptimizedSegmentor.load_pretrained(args.model_path, device=device)
    model.to(device)
    model.eval()
    
    print(f"\nPreparing {args.split} dataset with precomputed bbox extraction...")
    
    data_root = data_cfg['data_root']
    target_size = training_cfg['image_size']
    bbox_cfg = config.get('bbox', {})
    
    csv_file = data_cfg['val_csv'] if args.split == 'val' else data_cfg['train_csv']
    detection_csv = data_cfg['val_detection_csv'] if args.split == 'val' else data_cfg['train_detection_csv']
    
    dataset = SparkSegmentationOptimizedDataset(
        csv_path=os.path.join(data_root, csv_file),
        image_root=os.path.join(data_root, data_cfg['image_root']),
        mask_root=os.path.join(data_root, data_cfg['mask_root']),
        split=args.split,
        detection_csv=detection_csv,
        target_size=target_size,
        bbox_expansion=bbox_cfg.get('expansion', 1.1),
        fallback_mode=bbox_cfg.get('fallback_mode', 'full_image'),
        augment=False
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn_segmentation_optimized
    )
    
    print(f"Evaluating on {len(dataset)} samples...")
    
    # Prepare paths for full-image loading
    image_root_full = os.path.join(data_root, data_cfg['image_root'])
    mask_root_full = os.path.join(data_root, data_cfg['mask_root'])
    
    results = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        save_examples=args.save_examples,
        save_path=output_dir,
        use_fp16=not args.no_fp16,
        dataset=dataset,
        data_root=data_root,
        image_root=image_root_full,
        mask_root=mask_root_full,
        split=args.split
    )
    
    print_evaluation_results(results)
    save_results(results, output_dir)
    plot_confusion_matrix(results['confusion_matrix'], output_dir)
    
    # Plot bbox distribution if we have the data
    bbox_size_stats = results.get('bbox_size_stats', {})
    if bbox_size_stats and args.save_examples:
        # Collect bbox sizes from all samples
        bbox_sizes = []
        for i in range(len(dataset)):
            sample = dataset[i]
            meta = sample['metadata']
            bbox_sizes.append((meta['bbox_width'], meta['bbox_height']))
        plot_bbox_distribution(bbox_sizes, output_dir, target_size)
    
    print(f"\n✅ Evaluation complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
