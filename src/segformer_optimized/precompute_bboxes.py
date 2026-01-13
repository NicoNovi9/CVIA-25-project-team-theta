"""
Precompute bounding boxes for SegFormer Optimized training.

This script uses a trained YOLO model to detect spacecraft bounding boxes
in the SPARK dataset and saves them to a CSV file for use during training.

Output CSV format:
    Image name,Class,Bounding box
    image001.jpg,VenusExpress,"(x_min, y_min, x_max, y_max)"

Usage:
    python precompute_bboxes.py \
        --model_path trained_models/model_weights_yolo/yolo11n_640/weights/best.pt \
        --data_root /project/scratch/p200981/spark2024 \
        --split train \
        --output_csv train_detection.csv
"""

import os
import sys
import argparse
import csv
import time
from pathlib import Path
from tqdm import tqdm

import torch
import numpy as np
import pandas as pd
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute YOLO bounding boxes for SegFormer Optimized")
    
    # Model path
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to YOLO model weights (.pt file)")
    
    # Data paths
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory of SPARK dataset")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val"],
                        help="Dataset split to process (train or val)")
    
    # Output
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Output CSV filename (default: {split}_detection.csv)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: same as data_root)")
    
    # YOLO inference settings
    parser.add_argument("--confidence", type=float, default=0.25,
                        help="Confidence threshold for detection")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="IoU threshold for NMS")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size for YOLO inference")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (e.g., '0', 'cpu')")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Use FP16 for faster inference")
    
    return parser.parse_args()


def load_dataset_info(data_root, split):
    """Load dataset CSV file."""
    csv_path = os.path.join(data_root, f"{split}.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")
    
    return df


def get_image_paths(df, data_root, split):
    """Get list of all image paths from dataset."""
    image_paths = []
    
    for idx in range(len(df)):
        row = df.iloc[idx]
        img_path = os.path.join(
            data_root, "images", row["Class"], split, row["Image name"]
        )
        
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue
        
        image_paths.append({
            'path': img_path,
            'image_name': row["Image name"],
            'class': row["Class"]
        })
    
    return image_paths


def run_yolo_detection(model, image_infos, args):
    """
    Run YOLO detection on all images.
    
    Returns:
        List of detection results with format:
        {
            'image_name': str,
            'class': str,
            'bbox': tuple (x_min, y_min, x_max, y_max) or None
        }
    """
    results = []
    
    print(f"\nRunning YOLO detection on {len(image_infos)} images...")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.imgsz}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"Device: {args.device}")
    
    start_time = time.time()
    
    # Process in batches
    batch_size = args.batch_size
    
    for i in tqdm(range(0, len(image_infos), batch_size), desc="Detection"):
        batch_infos = image_infos[i:i+batch_size]
        batch_paths = [info['path'] for info in batch_infos]
        
        # Run prediction
        predictions = model.predict(
            source=batch_paths,
            imgsz=args.imgsz,
            conf=args.confidence,
            iou=args.iou,
            device=args.device,
            half=args.fp16 and torch.cuda.is_available(),
            verbose=False,
        )
        
        # Process each prediction
        for img_info, pred in zip(batch_infos, predictions):
            # Get best detection (highest confidence)
            if len(pred.boxes) > 0:
                # Get highest confidence detection
                best_idx = pred.boxes.conf.argmax()
                
                # Get bounding box (xyxy format)
                box = pred.boxes.xyxy[best_idx].cpu().numpy()
                x_min, y_min, x_max, y_max = box
                
                bbox = (int(round(x_min)), int(round(y_min)), 
                       int(round(x_max)), int(round(y_max)))
            else:
                # No detection found
                bbox = None
            
            results.append({
                'image_name': img_info['image_name'],
                'class': img_info['class'],
                'bbox': bbox
            })
    
    elapsed_time = time.time() - start_time
    fps = len(image_infos) / elapsed_time
    
    print(f"\nDetection completed in {elapsed_time:.2f}s ({fps:.1f} FPS)")
    
    # Print statistics
    detected = sum(1 for r in results if r['bbox'] is not None)
    print(f"Detections found: {detected}/{len(results)} ({100*detected/len(results):.1f}%)")
    
    return results


def save_detection_csv(results, output_path):
    """
    Save detection results to CSV file.
    
    Format:
        Image name,Class,Bounding box
        image001.jpg,VenusExpress,"(x_min, y_min, x_max, y_max)"
    
    For images with no detection, we save an empty bbox string.
    """
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image name', 'Class', 'Bounding box'])
        
        for r in results:
            if r['bbox'] is not None:
                x_min, y_min, x_max, y_max = r['bbox']
                bbox_str = f"({x_min}, {y_min}, {x_max}, {y_max})"
            else:
                bbox_str = ""  # Empty string for no detection
            
            writer.writerow([r['image_name'], r['class'], bbox_str])
    
    print(f"\n✓ Detection CSV saved to: {output_path}")


def main():
    args = parse_args()
    
    print("=" * 80)
    print("YOLO BOUNDING BOX PRECOMPUTATION FOR SEGFORMER OPTIMIZED")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Data root: {args.data_root}")
    print(f"Split: {args.split}")
    print("=" * 80 + "\n")
    
    # Determine output path
    if args.output_dir is None:
        args.output_dir = args.data_root
    
    if args.output_csv is None:
        args.output_csv = f"{args.split}_detection.csv"
    
    output_path = os.path.join(args.output_dir, args.output_csv)
    
    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine device
    if args.device is None:
        args.device = 0 if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {args.device}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print()
    
    # ==========================================================================
    # LOAD MODEL
    # ==========================================================================
    print("[1/4] Loading YOLO model...")
    
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)
    
    if not os.path.exists(args.model_path):
        print(f"[ERROR] Model not found: {args.model_path}")
        sys.exit(1)
    
    model = YOLO(args.model_path)
    print(f"✓ Model loaded from {args.model_path}\n")
    
    # ==========================================================================
    # LOAD DATASET INFO
    # ==========================================================================
    print("[2/4] Loading dataset info...")
    
    df = load_dataset_info(args.data_root, args.split)
    image_infos = get_image_paths(df, args.data_root, args.split)
    
    if len(image_infos) == 0:
        print("[ERROR] No images found!")
        sys.exit(1)
    
    print(f"✓ Found {len(image_infos)} images\n")
    
    # ==========================================================================
    # RUN DETECTION
    # ==========================================================================
    print("[3/4] Running YOLO detection...")
    
    results = run_yolo_detection(model, image_infos, args)
    
    # ==========================================================================
    # SAVE RESULTS
    # ==========================================================================
    print("\n[4/4] Saving results...")
    
    save_detection_csv(results, output_path)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total images processed: {len(results)}")
    
    # Detection statistics
    detected = sum(1 for r in results if r['bbox'] is not None)
    no_detection = len(results) - detected
    
    print(f"\nDetection results:")
    print(f"  With detection:    {detected:6d} ({100*detected/len(results):5.1f}%)")
    print(f"  Without detection: {no_detection:6d} ({100*no_detection/len(results):5.1f}%)")
    
    # Count per-class detections
    class_counts = {}
    for r in results:
        cls = r['class']
        class_counts[cls] = class_counts.get(cls, 0) + 1
    
    print(f"\nSamples per class:")
    for cls in sorted(class_counts.keys()):
        print(f"  {cls}: {class_counts[cls]}")
    
    # Bounding box size statistics (for detected boxes)
    if detected > 0:
        bbox_sizes = []
        for r in results:
            if r['bbox'] is not None:
                x_min, y_min, x_max, y_max = r['bbox']
                width = x_max - x_min
                height = y_max - y_min
                bbox_sizes.append((width, height))
        
        widths = [w for w, h in bbox_sizes]
        heights = [h for w, h in bbox_sizes]
        areas = [w * h for w, h in bbox_sizes]
        
        print(f"\nBounding box statistics:")
        print(f"  Width:  mean={np.mean(widths):.1f}, median={np.median(widths):.1f}, "
              f"min={np.min(widths)}, max={np.max(widths)}")
        print(f"  Height: mean={np.mean(heights):.1f}, median={np.median(heights):.1f}, "
              f"min={np.min(heights)}, max={np.max(heights)}")
        print(f"  Area:   mean={np.mean(areas):.0f}, median={np.median(areas):.0f}")
        
        # Count how many are smaller than 512x512
        smaller_than_512 = sum(1 for w, h in bbox_sizes if w <= 512 and h <= 512)
        print(f"\n  BBoxes ≤ 512x512: {smaller_than_512}/{len(bbox_sizes)} "
              f"({100*smaller_than_512/len(bbox_sizes):.1f}%)")
    
    print("\n" + "=" * 80)
    print("PRECOMPUTATION COMPLETED")
    print("=" * 80)
    print(f"Output CSV: {output_path}")
    print("\nYou can now use this CSV for training SegFormer Optimized:")
    print(f"  detection_csv: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
