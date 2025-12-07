"""
Fast Multi-GPU Inference script for DETR detection model.
Generates detection.csv for competition submission.

Submission format:
submission.zip
├── detection.csv       <-- Detection Results (Required Name)

Usage:
    python inference_detr.py \
        --model_path model_weights_DETR_30epochs/detr_best \
        --data_path /project/scratch/p200981/spark2024_test/detection/images \
        --output_dir submission_output
"""

import os
import sys
import argparse
import glob
import csv
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="DETR Detection Inference")
    
    # Model path
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to DETR detection model directory")
    
    # Data path
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to detection test images directory")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="submission_output",
                        help="Directory to save submission files")
    
    # Inference settings
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Confidence threshold for detection")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference (per GPU)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of data loading workers")
    
    # Flags
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Use FP16 for faster inference")
    
    return parser.parse_args()


# =============================================================================
# DATASET
# =============================================================================

class DetectionDataset(Dataset):
    """Dataset for detection inference."""
    def __init__(self, image_paths, processor):
        self.image_paths = image_paths
        self.processor = processor
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        orig_size = img.size  # (W, H)
        
        # Process image
        inputs = self.processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        
        return {
            "pixel_values": pixel_values,
            "filename": os.path.basename(img_path),
            "orig_w": orig_size[0],
            "orig_h": orig_size[1],
        }


def detection_collate_fn(batch):
    """Collate function for detection dataset."""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    filenames = [item["filename"] for item in batch]
    orig_ws = [item["orig_w"] for item in batch]
    orig_hs = [item["orig_h"] for item in batch]
    
    return {
        "pixel_values": pixel_values,
        "filenames": filenames,
        "orig_ws": orig_ws,
        "orig_hs": orig_hs,
    }


# =============================================================================
# INFERENCE FUNCTIONS
# =============================================================================

def get_test_images(data_path):
    """Get list of test images from directory."""
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(data_path, ext)))
    images = sorted(images)
    return images


def run_detection_inference(model, processor, id2label, image_paths, device, 
                            threshold=0.3, batch_size=32, num_workers=8, use_fp16=True):
    """
    Run detection inference using DataLoader for efficient batching.
    """
    print(f"\n[Detection] Running inference on {len(image_paths)} images...")
    print(f"[Detection] Batch size: {batch_size}, Workers: {num_workers}")
    
    # Create dataset and dataloader
    dataset = DetectionDataset(image_paths, processor)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=detection_collate_fn,
        prefetch_factor=2,
    )
    
    results = []
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Detection")):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            filenames = batch["filenames"]
            orig_ws = batch["orig_ws"]
            orig_hs = batch["orig_hs"]
            
            # Forward pass with FP16
            if use_fp16:
                with torch.cuda.amp.autocast():
                    outputs = model(pixel_values=pixel_values)
            else:
                outputs = model(pixel_values=pixel_values)
            
            # Post-process each image in batch
            for j in range(len(filenames)):
                orig_w, orig_h = orig_ws[j], orig_hs[j]
                filename = filenames[j]
                
                # Get outputs for this image
                single_outputs = type(outputs)(
                    logits=outputs.logits[j:j+1],
                    pred_boxes=outputs.pred_boxes[j:j+1]
                )
                
                # Post-process to original image size
                target_sizes = torch.tensor([[orig_h, orig_w]]).to(device)
                processed = processor.post_process_object_detection(
                    single_outputs,
                    target_sizes=target_sizes,
                    threshold=0.0  # Get all, filter later
                )[0]
                
                # Get best prediction
                if len(processed["scores"]) > 0:
                    # Filter by threshold first
                    mask = processed["scores"] >= threshold
                    if mask.any():
                        filtered_scores = processed["scores"][mask]
                        filtered_labels = processed["labels"][mask]
                        filtered_boxes = processed["boxes"][mask]
                        best_idx = filtered_scores.argmax()
                        score = filtered_scores[best_idx].item()
                        label_id = filtered_labels[best_idx].item()
                        box = filtered_boxes[best_idx].tolist()
                    else:
                        # Use best even below threshold
                        best_idx = processed["scores"].argmax()
                        score = processed["scores"][best_idx].item()
                        label_id = processed["labels"][best_idx].item()
                        box = processed["boxes"][best_idx].tolist()
                    
                    x_min, y_min, x_max, y_max = box
                    bbox_str = f"({int(round(x_min))}, {int(round(y_min))}, {int(round(x_max))}, {int(round(y_max))})"
                    class_name = id2label.get(label_id, f"class_{label_id}")
                else:
                    # Fallback
                    cx, cy = orig_w // 2, orig_h // 2
                    w, h = orig_w // 4, orig_h // 4
                    bbox_str = f"({cx - w//2}, {cy - h//2}, {cx + w//2}, {cy + h//2})"
                    class_name = list(id2label.values())[0]
                    score = 0.0
                
                results.append({
                    "filename": filename,
                    "class": class_name,
                    "bbox": bbox_str,
                    "score": score
                })
            
            # Log progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                imgs_done = (batch_idx + 1) * batch_size
                rate = imgs_done / elapsed
                print(f"  [Detection] Processed {imgs_done}/{len(image_paths)} images ({rate:.1f} img/s)")
    
    elapsed = time.time() - start_time
    print(f"[Detection] Completed in {elapsed:.1f}s ({len(image_paths)/elapsed:.1f} img/s)")
    
    return results


def save_detection_csv(results, output_path):
    """Save detection results to CSV."""
    print(f"\nSaving detection.csv to {output_path}...")
    
    # Sort by filename
    results_sorted = sorted(results, key=lambda x: x['filename'])
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'class', 'bbox'])
        for result in results_sorted:
            writer.writerow([result['filename'], result['class'], result['bbox']])
    
    print(f"  Saved {len(results_sorted)} entries")


def main():
    args = parse_args()
    
    # =========================================================================
    # SETUP
    # =========================================================================
    print("=" * 70)
    print("DETR DETECTION INFERENCE")
    print("=" * 70)
    
    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    total_start = time.time()
    
    # =========================================================================
    # DETECTION TASK
    # =========================================================================
    print("\n" + "=" * 70)
    print("SPACECRAFT DETECTION")
    print("=" * 70)
    
    # Load detection model
    print(f"Loading detection model from {args.model_path}...")
    detection_model = DetrForObjectDetection.from_pretrained(args.model_path)
    processor = DetrImageProcessor.from_pretrained(args.model_path)
    
    # Use DataParallel if multiple GPUs
    if num_gpus > 1:
        print(f"Using DataParallel across {num_gpus} GPUs")
        detection_model = nn.DataParallel(detection_model)
    
    detection_model.to(device)
    detection_model.eval()
    
    # Get label mappings (handle DataParallel wrapper)
    base_model = detection_model.module if hasattr(detection_model, 'module') else detection_model
    id2label = {int(k): v for k, v in base_model.config.id2label.items()}
    print(f"Classes: {id2label}")
    
    # Get test images
    detection_images = get_test_images(args.data_path)
    print(f"Found {len(detection_images)} detection test images")
    
    # Run detection - adjust batch size for multi-GPU
    effective_batch = args.batch_size * max(1, num_gpus)
    detection_results = run_detection_inference(
        model=detection_model,
        processor=processor,
        id2label=id2label,
        image_paths=detection_images,
        device=device,
        threshold=args.threshold,
        batch_size=effective_batch,
        num_workers=args.num_workers,
        use_fp16=args.fp16
    )
    
    # Save detection CSV
    save_detection_csv(detection_results, os.path.join(args.output_dir, 'detection.csv'))
    
    # Free memory
    del detection_model
    torch.cuda.empty_cache()
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    total_time = time.time() - total_start
    print("\n" + "=" * 70)
    print("DETECTION INFERENCE COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Output directory: {args.output_dir}")
    print(f"  - detection.csv: {len(detection_results)} entries")
    
    # Sample outputs
    print("\nSample detection results:")
    for r in sorted(detection_results, key=lambda x: x['filename'])[:5]:
        print(f"  {r['filename']}: {r['class']} @ {r['bbox']} (score={r['score']:.3f})")


if __name__ == "__main__":
    main()
