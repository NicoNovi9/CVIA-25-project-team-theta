"""
Inference script for SegFormer Optimized segmentation model.
Generates NPZ segmentation masks for competition submission.

Full inference pipeline:
1. YOLO detects spacecraft bounding box
2. Extract and prepare bbox region (pad/resize to 512x512)
3. Run SegFormer segmentation on bbox
4. Map prediction back to original image coordinates
5. All pixels outside bbox are classified as background
6. Save as NPZ boolean RGB mask

Submission format:
submission.zip
├── test_00000_layer.npz  <-- Segmentation Result
└── ...

Usage:
    python inference_segformer_optimized.py \
        --model_path model_weights_segformer_optimized/segformer_optimized_best \
        --data_path /project/scratch/p200981/spark2024_test/segmentation/stream-1-test \
        --output_dir submission_output
"""

import os
import sys
import argparse
import glob
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import yaml

from segformer_optimized_model import SegFormerOptimizedSegmentor, BBoxProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="SegFormer Optimized Segmentation Inference")
    
    # Model
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to SegFormer Optimized model directory")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config file")
    
    # YOLO
    parser.add_argument("--yolo_model", type=str, required=True,
                        help="Path to YOLO model weights for detection")
    parser.add_argument("--yolo_conf", type=float, default=0.25,
                        help="YOLO confidence threshold")
    parser.add_argument("--yolo_iou", type=float, default=0.45,
                        help="YOLO IoU threshold for NMS")
    parser.add_argument("--yolo_imgsz", type=int, default=640,
                        help="YOLO image size")
    
    # Data
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to test images directory")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="submission_output",
                        help="Directory to save submission files")
    
    # Inference settings
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for SegFormer inference")
    parser.add_argument("--segformer_size", type=int, default=512,
                        help="Size for SegFormer input (after bbox extraction)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of data loading workers")
    parser.add_argument("--bbox_expansion", type=float, default=1.1,
                        help="BBox expansion factor")
    
    # Flags
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Use FP16 for faster inference")
    parser.add_argument("--no_fp16", action="store_false", dest="fp16",
                        help="Disable FP16")
    parser.add_argument("--tta", action="store_true",
                        help="Use test-time augmentation (flip)")
    parser.add_argument("--fallback_mode", type=str, default="full_image",
                        choices=["full_image", "center_crop"],
                        help="Fallback when no detection")
    
    return parser.parse_args()


# =============================================================================
# YOLO DETECTOR
# =============================================================================

class YOLOInferenceDetector:
    """YOLO detector for inference (no caching, batch processing)."""
    
    def __init__(self, model_path, conf=0.25, iou=0.45, imgsz=640, device=None):
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self._model = None
    
    def _load_model(self):
        if self._model is None:
            from ultralytics import YOLO
            print(f"[YOLO] Loading model from {self.model_path}")
            self._model = YOLO(self.model_path)
    
    def detect_batch(self, image_paths):
        """
        Detect bounding boxes for a batch of images.
        
        Returns:
            Dict mapping image_path -> (x_min, y_min, x_max, y_max) or None
        """
        self._load_model()
        
        results = self._model.predict(
            source=image_paths,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False
        )
        
        detections = {}
        for img_path, result in zip(image_paths, results):
            if len(result.boxes) > 0:
                best_idx = result.boxes.conf.argmax()
                box = result.boxes.xyxy[best_idx].cpu().numpy()
                detections[img_path] = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            else:
                detections[img_path] = None
        
        return detections


# =============================================================================
# DATASET
# =============================================================================

class SegmentationInferenceDataset(Dataset):
    """Dataset for inference that provides raw images."""
    
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        orig_size = img.size  # (W, H)
        
        return {
            "image_path": img_path,
            "filename": os.path.basename(img_path),
            "orig_w": orig_size[0],
            "orig_h": orig_size[1],
        }


# =============================================================================
# INFERENCE PIPELINE
# =============================================================================

def expand_bbox(bbox, expansion, img_width, img_height):
    """Expand bounding box by factor while staying in bounds."""
    x_min, y_min, x_max, y_max = bbox
    
    width = x_max - x_min
    height = y_max - y_min
    
    expand_w = int(width * (expansion - 1) / 2)
    expand_h = int(height * (expansion - 1) / 2)
    
    x_min = max(0, x_min - expand_w)
    y_min = max(0, y_min - expand_h)
    x_max = min(img_width, x_max + expand_w)
    y_max = min(img_height, y_max + expand_h)
    
    return (x_min, y_min, x_max, y_max)


def extract_and_prepare_bbox(image, bbox, target_size):
    """
    Extract bbox region and prepare for SegFormer.
    
    Args:
        image: PIL Image
        bbox: (x_min, y_min, x_max, y_max)
        target_size: Target size (512)
    
    Returns:
        prepared_tensor: [C, target_size, target_size]
        metadata: Dict with extraction info
    """
    x_min, y_min, x_max, y_max = bbox
    img_w, img_h = image.size
    
    # Clamp coordinates
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_w, x_max)
    y_max = min(img_h, y_max)
    
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    
    # Extract crop
    crop = image.crop((x_min, y_min, x_max, y_max))
    crop_tensor = TF.to_tensor(crop)  # [C, H, W]
    
    # Determine if padding or downsampling
    needs_padding = bbox_width <= target_size and bbox_height <= target_size
    
    if needs_padding:
        # Pad to target size
        pad_right = target_size - bbox_width
        pad_bottom = target_size - bbox_height
        
        prepared = F.pad(crop_tensor, (0, pad_right, 0, pad_bottom), value=0)
        scale_factor = 1.0
    else:
        # Downsample to target size
        prepared = F.interpolate(
            crop_tensor.unsqueeze(0),
            size=(target_size, target_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        scale_factor = min(target_size / bbox_width, target_size / bbox_height)
    
    # Normalize
    prepared = TF.normalize(prepared, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    metadata = {
        'bbox': (x_min, y_min, x_max, y_max),
        'bbox_width': bbox_width,
        'bbox_height': bbox_height,
        'needs_padding': needs_padding,
        'scale_factor': scale_factor,
        'orig_size': (img_h, img_w)
    }
    
    return prepared, metadata


def map_prediction_to_full_image(prediction, metadata, output_size):
    """
    Map bbox prediction back to full image coordinates.
    
    Args:
        prediction: Predicted mask [H, W] with class indices
        metadata: Dict from extract_and_prepare_bbox
        output_size: (H, W) of original image
    
    Returns:
        Full-size prediction [H, W]
    """
    x_min, y_min, x_max, y_max = metadata['bbox']
    bbox_width = metadata['bbox_width']
    bbox_height = metadata['bbox_height']
    needs_padding = metadata['needs_padding']
    
    if needs_padding:
        # Remove padding
        bbox_pred = prediction[:bbox_height, :bbox_width]
    else:
        # Upsample back to bbox size
        bbox_pred = F.interpolate(
            prediction.unsqueeze(0).unsqueeze(0).float(),
            size=(bbox_height, bbox_width),
            mode='nearest'
        ).squeeze(0).squeeze(0).long()
    
    # Create full-size output (background = 0)
    full_pred = torch.zeros(output_size[0], output_size[1], 
                           device=prediction.device, dtype=torch.long)
    
    # Place bbox prediction
    full_pred[y_min:y_max, x_min:x_max] = bbox_pred
    
    return full_pred


def get_test_images(data_path):
    """Get list of test images from directory."""
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(data_path, ext)))
    images = sorted(images)
    return images


def run_inference(model, yolo_detector, image_paths, args):
    """
    Run full inference pipeline.
    
    1. Run YOLO detection (in batches)
    2. Extract and prepare bbox regions
    3. Run SegFormer segmentation
    4. Map predictions back to full images
    5. Save as NPZ files
    """
    print(f"\n[Inference] Processing {len(image_paths)} images...")
    print(f"[Inference] SegFormer size: {args.segformer_size}")
    print(f"[Inference] BBox expansion: {args.bbox_expansion}")
    print(f"[Inference] FP16: {args.fp16}")
    print(f"[Inference] TTA: {args.tta}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Class to RGB color mapping
    class_to_rgb = {
        0: [0, 0, 0],       # background - black
        1: [255, 0, 0],     # spacecraft body - red
        2: [0, 0, 255],     # solar panels - blue
    }
    
    saved_files = []
    start_time = time.time()
    
    # Statistics
    stats = {'padded': 0, 'downsampled': 0, 'no_detection': 0}
    
    device = next(model.parameters()).device
    model.eval()
    
    # Process in batches
    batch_size = args.batch_size
    yolo_batch_size = min(64, batch_size * 2)  # YOLO can handle larger batches
    
    # First, run YOLO detection on all images
    print("\n[Step 1/3] Running YOLO detection...")
    all_detections = {}
    for i in tqdm(range(0, len(image_paths), yolo_batch_size), desc="YOLO Detection"):
        batch_paths = image_paths[i:i + yolo_batch_size]
        batch_detections = yolo_detector.detect_batch(batch_paths)
        all_detections.update(batch_detections)
    
    detected_count = sum(1 for d in all_detections.values() if d is not None)
    print(f"[YOLO] Detected: {detected_count}/{len(image_paths)} ({100*detected_count/len(image_paths):.1f}%)")
    
    # Now process with SegFormer
    print("\n[Step 2/3] Running SegFormer segmentation...")
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Segmentation"):
        batch_paths = image_paths[i:i + batch_size]
        
        # Prepare batch
        batch_images = []
        batch_metadatas = []
        batch_orig_images = []
        
        for img_path in batch_paths:
            img = Image.open(img_path).convert("RGB")
            orig_w, orig_h = img.size
            
            # Get detection
            bbox = all_detections.get(img_path)
            
            # Handle no detection
            if bbox is None:
                stats['no_detection'] += 1
                if args.fallback_mode == 'full_image':
                    bbox = (0, 0, orig_w, orig_h)
                else:  # center_crop
                    center_x, center_y = orig_w // 2, orig_h // 2
                    crop_size = min(orig_w, orig_h, args.segformer_size)
                    half_size = crop_size // 2
                    bbox = (center_x - half_size, center_y - half_size,
                           center_x + half_size, center_y + half_size)
            
            # Expand bbox
            bbox = expand_bbox(bbox, args.bbox_expansion, orig_w, orig_h)
            
            # Extract and prepare
            prepared, metadata = extract_and_prepare_bbox(img, bbox, args.segformer_size)
            
            # Track statistics
            if metadata['needs_padding']:
                stats['padded'] += 1
            else:
                stats['downsampled'] += 1
            
            batch_images.append(prepared)
            batch_metadatas.append(metadata)
            batch_orig_images.append(img)
        
        # Stack batch
        images_batch = torch.stack(batch_images).to(device)
        
        # Forward pass
        with torch.no_grad():
            if args.fp16:
                with torch.cuda.amp.autocast():
                    outputs = model(images_batch)
                    logits = outputs['logits']
                    
                    if args.tta:
                        images_flip = torch.flip(images_batch, dims=[-1])
                        outputs_flip = model(images_flip)
                        logits_flip = torch.flip(outputs_flip['logits'], dims=[-1])
                        logits = (logits + logits_flip) / 2
            else:
                outputs = model(images_batch)
                logits = outputs['logits']
                
                if args.tta:
                    images_flip = torch.flip(images_batch, dims=[-1])
                    outputs_flip = model(images_flip)
                    logits_flip = torch.flip(outputs_flip['logits'], dims=[-1])
                    logits = (logits + logits_flip) / 2
        
        # Get predictions
        pred_masks = torch.argmax(logits, dim=1)
        
        # Process each prediction
        for j, img_path in enumerate(batch_paths):
            filename = os.path.basename(img_path)
            metadata = batch_metadatas[j]
            orig_h, orig_w = metadata['orig_size']
            
            # Map prediction to full image
            full_pred = map_prediction_to_full_image(
                pred_masks[j],
                metadata,
                (orig_h, orig_w)
            )
            
            # Convert to RGB
            class_mask = full_pred.cpu().numpy().astype(np.uint8)
            H, W = class_mask.shape
            rgb_mask = np.zeros((H, W, 3), dtype=np.uint8)
            for class_idx, color in class_to_rgb.items():
                rgb_mask[class_mask == class_idx] = color
            
            # Convert to boolean
            rgb_bool = rgb_mask > 127
            
            # Save as NPZ
            base_name = filename.rsplit('_', 1)[0]  # test_00000
            npz_name = f"{base_name}_layer.npz"
            npz_path = os.path.join(args.output_dir, npz_name)
            np.savez_compressed(npz_path, data=rgb_bool)
            saved_files.append(npz_name)
    
    elapsed = time.time() - start_time
    print(f"\n[Inference] Completed in {elapsed:.1f}s ({len(image_paths)/elapsed:.1f} img/s)")
    
    # Print statistics
    total = stats['padded'] + stats['downsampled']
    print(f"\n[Statistics]")
    print(f"  Padded (preserved resolution): {stats['padded']} ({100*stats['padded']/total:.1f}%)")
    print(f"  Downsampled: {stats['downsampled']} ({100*stats['downsampled']/total:.1f}%)")
    print(f"  No YOLO detection (used fallback): {stats['no_detection']}")
    
    return saved_files


def main():
    args = parse_args()
    
    print("=" * 70)
    print("SEGFORMER OPTIMIZED INFERENCE (YOLO-guided)")
    print("=" * 70)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Available GPUs: {num_gpus}")
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Load YOLO detector
    print(f"\nLoading YOLO from {args.yolo_model}...")
    yolo_detector = YOLOInferenceDetector(
        model_path=args.yolo_model,
        conf=args.yolo_conf,
        iou=args.yolo_iou,
        imgsz=args.yolo_imgsz,
        device=device if torch.cuda.is_available() else 'cpu'
    )
    
    # Load SegFormer model
    print(f"\nLoading SegFormer from {args.model_path}...")
    model = SegFormerOptimizedSegmentor.load_pretrained(args.model_path, device=device)
    
    # Use DataParallel for multi-GPU
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"Using DataParallel across {num_gpus} GPUs")
        model = nn.DataParallel(model)
    
    model.to(device)
    model.eval()
    
    # Get test images
    image_paths = get_test_images(args.data_path)
    print(f"\nFound {len(image_paths)} test images in {args.data_path}")
    
    if len(image_paths) == 0:
        print("ERROR: No images found!")
        return
    
    # Run inference
    saved_files = run_inference(model, yolo_detector, image_paths, args)
    
    print(f"\n✅ Inference complete!")
    print(f"   Saved {len(saved_files)} NPZ files to {args.output_dir}")
    
    # Print sample files
    print("\nSample output files:")
    for f in saved_files[:5]:
        print(f"  - {f}")
    if len(saved_files) > 5:
        print(f"  ... and {len(saved_files) - 5} more")


if __name__ == "__main__":
    main()
