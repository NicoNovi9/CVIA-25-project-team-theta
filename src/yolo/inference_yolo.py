"""
Inference script for YOLO detection model.
Generates detection.csv for competition submission.

Submission format:
submission.zip
├── detection.csv       <-- Detection Results (Required Name)

Usage:
    python inference_yolo.py \
        --model_path model_weights_yolo/yolo11n_640/weights/best.pt \
        --data_path /project/scratch/p200981/spark2024_test/detection/images \
        --output_dir submission_output
"""

import os
import sys
import argparse
import glob
import csv
import time
import zipfile
from pathlib import Path
from tqdm import tqdm

import torch
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset_yolo import CLASS_NAMES, ID_TO_CLASS


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Detection Inference")
    
    # Model path
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to YOLO model weights (.pt file)")
    
    # Data path
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to detection test images directory")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="submission_output",
                        help="Directory to save submission files")
    
    # Inference settings
    parser.add_argument("--threshold", type=float, default=0.25,
                        help="Confidence threshold for detection")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="IoU threshold for NMS")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size for inference")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (e.g., '0', '0,1,2,3', 'cpu')")
    
    # Flags
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Use FP16 for faster inference")
    parser.add_argument("--save_visualizations", action="store_true",
                        help="Save annotated images")
    
    return parser.parse_args()


def get_test_images(data_path):
    """Get list of test images from directory."""
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    images = []
    
    data_path = Path(data_path)
    
    for ext in extensions:
        images.extend(data_path.glob(ext))
    
    images = sorted(images)
    return images


def run_inference(model, image_paths, args, id2label):
    """
    Run detection inference on images.
    
    Args:
        model: YOLO model
        image_paths: List of image paths
        args: Command line arguments
        id2label: Class ID to label mapping
    
    Returns:
        List of detection results
    """
    results = []
    
    print(f"\n[Inference] Processing {len(image_paths)} images...")
    print(f"[Inference] Batch size: {args.batch_size}")
    print(f"[Inference] Image size: {args.imgsz}")
    print(f"[Inference] Confidence threshold: {args.threshold}")
    
    start_time = time.time()
    
    # Process in batches
    batch_size = args.batch_size
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Detection"):
        batch_paths = image_paths[i:i+batch_size]
        
        # Run prediction
        predictions = model.predict(
            source=[str(p) for p in batch_paths],
            imgsz=args.imgsz,
            conf=args.threshold,
            iou=args.iou,
            device=args.device,
            half=args.fp16 and torch.cuda.is_available(),
            verbose=False,
            save=args.save_visualizations,
        )
        
        # Process each prediction
        for img_path, pred in zip(batch_paths, predictions):
            filename = img_path.name
            
            # Get best detection (highest confidence)
            if len(pred.boxes) > 0:
                # Get highest confidence detection
                best_idx = pred.boxes.conf.argmax()
                
                # Get class and confidence
                class_id = int(pred.boxes.cls[best_idx])
                confidence = float(pred.boxes.conf[best_idx])
                
                # Get bounding box (xyxy format)
                box = pred.boxes.xyxy[best_idx].cpu().numpy()
                x_min, y_min, x_max, y_max = box
                
                # Get class name
                class_name = id2label.get(class_id, CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}")
                
                # Format bounding box string
                bbox_str = f"({int(round(x_min))}, {int(round(y_min))}, {int(round(x_max))}, {int(round(y_max))})"
                
            else:
                # No detection - use default (should not happen often)
                class_name = CLASS_NAMES[0]  # Default to first class
                bbox_str = "(0, 0, 100, 100)"  # Default bbox
                confidence = 0.0
            
            results.append({
                'filename': filename,
                'class': class_name,
                'bbox': bbox_str,
                'confidence': confidence
            })
    
    elapsed_time = time.time() - start_time
    fps = len(image_paths) / elapsed_time
    
    print(f"\n[Inference] Completed in {elapsed_time:.2f}s ({fps:.1f} FPS)")
    
    return results


def save_detection_csv(results, output_path):
    """
    Save detection results to CSV file.
    
    Expected format:
        Image name,Class,Bounding box
        image001.jpg,VenusExpress,"(x_min, y_min, x_max, y_max)"
    """
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'class', 'bbox'])
        
        for r in results:
            writer.writerow([r['filename'], r['class'], r['bbox']])
    
    print(f"[Output] Detection CSV saved to: {output_path}")


def create_submission_zip(output_dir, csv_filename='detection.csv'):
    """Create submission zip file."""
    output_dir = Path(output_dir)
    csv_path = output_dir / csv_filename
    zip_path = output_dir / "detection_submission.zip"
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, csv_filename)
    
    print(f"[Output] Submission zip created: {zip_path}")
    return zip_path


def main():
    args = parse_args()
    
    print("=" * 70)
    print("YOLO DETECTION INFERENCE")
    print("=" * 70)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_path}")
    print(f"Output: {args.output_dir}")
    print("=" * 70 + "\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = 0
    else:
        device = 'cpu'
    
    args.device = device
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print()
    
    # ==========================================================================
    # LOAD MODEL
    # ==========================================================================
    print("[1/3] Loading model...")
    
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)
    
    model = YOLO(args.model_path)
    
    # Get class names from model if available
    if hasattr(model, 'names') and model.names:
        id2label = model.names
    else:
        id2label = ID_TO_CLASS
    
    print(f"✓ Model loaded")
    print(f"Classes: {list(id2label.values())}")
    print()
    
    # ==========================================================================
    # GET TEST IMAGES
    # ==========================================================================
    print("[2/3] Loading test images...")
    
    image_paths = get_test_images(args.data_path)
    
    if len(image_paths) == 0:
        print(f"[ERROR] No images found in {args.data_path}")
        sys.exit(1)
    
    print(f"✓ Found {len(image_paths)} test images")
    print()
    
    # ==========================================================================
    # RUN INFERENCE
    # ==========================================================================
    print("[3/3] Running inference...")
    
    results = run_inference(model, image_paths, args, id2label)
    
    # ==========================================================================
    # SAVE RESULTS
    # ==========================================================================
    print("\n" + "-" * 70)
    print("SAVING RESULTS")
    print("-" * 70)
    
    # Save detection CSV
    csv_path = output_dir / "detection.csv"
    save_detection_csv(results, csv_path)
    
    # Create submission zip
    zip_path = create_submission_zip(output_dir)
    
    # Print summary
    print("\n" + "-" * 70)
    print("INFERENCE SUMMARY")
    print("-" * 70)
    print(f"Total images processed: {len(results)}")
    
    # Count per-class detections
    class_counts = {}
    for r in results:
        cls = r['class']
        class_counts[cls] = class_counts.get(cls, 0) + 1
    
    print("\nDetections per class:")
    for cls in sorted(class_counts.keys()):
        print(f"  {cls}: {class_counts[cls]}")
    
    # Confidence statistics
    confidences = [r['confidence'] for r in results]
    print(f"\nConfidence statistics:")
    print(f"  Mean: {np.mean(confidences):.4f}")
    print(f"  Min:  {np.min(confidences):.4f}")
    print(f"  Max:  {np.max(confidences):.4f}")
    
    print("\n" + "=" * 70)
    print("INFERENCE COMPLETED")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Detection CSV: {csv_path}")
    print(f"Submission ZIP: {zip_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
