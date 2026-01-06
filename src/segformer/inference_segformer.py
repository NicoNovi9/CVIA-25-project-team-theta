"""
Inference script for SegFormer segmentation model.
Generates NPZ segmentation masks for competition submission.

Submission format:
submission.zip
├── test_00000_layer.npz  <-- Segmentation Result
└── ...

Usage:
    python inference_segformer.py \
        --model_path model_weights_segformer/segformer_best \
        --data_path /project/scratch/p200981/spark2024_test/segmentation/stream-1-test \
        --output_dir submission_output

The output NPZ files contain boolean RGB masks where:
    - Red channel (R): spacecraft body
    - Blue channel (B): solar panels
    - Green channel (G): not used (background is absence of R and B)
"""

import os
import sys
import argparse
import glob
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import yaml

from segformer_model import SegFormerSegmentor


def parse_args():
    parser = argparse.ArgumentParser(description="SegFormer Segmentation Inference")
    
    # Model
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to SegFormer model directory")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config file (optional, will use model config if not provided)")
    
    # Data
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to test images directory")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="submission_output",
                        help="Directory to save submission files")
    
    # Inference settings
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument("--image_size", type=int, default=512,
                        help="Size to resize images for segmentation")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of data loading workers")
    
    # Flags
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Use FP16 for faster inference")
    parser.add_argument("--no_fp16", action="store_false", dest="fp16",
                        help="Disable FP16")
    parser.add_argument("--tta", action="store_true",
                        help="Use test-time augmentation (flip)")
    
    return parser.parse_args()


# =============================================================================
# DATASET
# =============================================================================

class SegmentationInferenceDataset(Dataset):
    """Dataset for segmentation inference."""
    
    def __init__(self, image_paths, target_size=512):
        self.image_paths = image_paths
        self.target_size = target_size
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.to_tensor = transforms.ToTensor()
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        orig_size = img.size  # (W, H)
        
        # Resize and normalize
        img_resized = img.resize((self.target_size, self.target_size), Image.BILINEAR)
        img_tensor = self.to_tensor(img_resized)
        img_tensor = self.normalize(img_tensor)
        
        return {
            "image": img_tensor,
            "filename": os.path.basename(img_path),
            "orig_w": orig_size[0],
            "orig_h": orig_size[1],
        }


def collate_fn(batch):
    """Collate function for inference."""
    images = torch.stack([item["image"] for item in batch])
    filenames = [item["filename"] for item in batch]
    orig_ws = [item["orig_w"] for item in batch]
    orig_hs = [item["orig_h"] for item in batch]
    
    return {
        "images": images,
        "filenames": filenames,
        "orig_ws": orig_ws,
        "orig_hs": orig_hs,
    }


# =============================================================================
# INFERENCE
# =============================================================================

def get_test_images(data_path):
    """Get list of test images from directory."""
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(data_path, ext)))
    images = sorted(images)
    return images


def run_inference(model, image_paths, device, output_dir, 
                  target_size=512, batch_size=32, num_workers=8, 
                  use_fp16=True, use_tta=False):
    """
    Run segmentation inference.
    
    Args:
        model: SegFormer model
        image_paths: List of image paths
        device: Device for inference
        output_dir: Output directory for NPZ files
        target_size: Image size for inference
        batch_size: Batch size
        num_workers: DataLoader workers
        use_fp16: Use FP16 for inference
        use_tta: Use test-time augmentation
    
    Returns:
        List of saved file names
    """
    print(f"\n[Segmentation] Running inference on {len(image_paths)} images...")
    print(f"[Segmentation] Batch size: {batch_size}, Workers: {num_workers}")
    print(f"[Segmentation] Image size: {target_size}")
    print(f"[Segmentation] FP16: {use_fp16}, TTA: {use_tta}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset and dataloader
    dataset = SegmentationInferenceDataset(image_paths, target_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        prefetch_factor=2,
    )
    
    saved_files = []
    start_time = time.time()
    
    # Class to RGB color mapping
    class_to_rgb = {
        0: [0, 0, 0],       # background - black
        1: [255, 0, 0],     # spacecraft body - red
        2: [0, 0, 255],     # solar panels - blue
    }
    
    model.eval()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Segmentation")):
            images = batch["images"].to(device, non_blocking=True)
            filenames = batch["filenames"]
            orig_ws = batch["orig_ws"]
            orig_hs = batch["orig_hs"]
            
            # Forward pass
            if use_fp16:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    logits = outputs['logits']  # [B, C, H, W]
                    
                    # Test-time augmentation: horizontal flip
                    if use_tta:
                        images_flip = torch.flip(images, dims=[-1])
                        outputs_flip = model(images_flip)
                        logits_flip = torch.flip(outputs_flip['logits'], dims=[-1])
                        logits = (logits + logits_flip) / 2
            else:
                outputs = model(images)
                logits = outputs['logits']
                
                if use_tta:
                    images_flip = torch.flip(images, dims=[-1])
                    outputs_flip = model(images_flip)
                    logits_flip = torch.flip(outputs_flip['logits'], dims=[-1])
                    logits = (logits + logits_flip) / 2
            
            # Get predicted masks
            pred_masks = torch.argmax(logits, dim=1)  # [B, H, W]
            
            # Save each mask
            for j in range(len(filenames)):
                filename = filenames[j]
                orig_w, orig_h = orig_ws[j], orig_hs[j]
                
                # Get class mask
                class_mask = pred_masks[j].cpu().numpy().astype(np.uint8)
                
                # Convert class indices to RGB image
                H, W = class_mask.shape
                rgb_mask = np.zeros((H, W, 3), dtype=np.uint8)
                for class_idx, color in class_to_rgb.items():
                    rgb_mask[class_mask == class_idx] = color
                
                # Resize RGB mask to original size
                rgb_pil = Image.fromarray(rgb_mask, mode='RGB')
                rgb_resized = rgb_pil.resize((orig_w, orig_h), Image.NEAREST)
                
                # Convert to boolean RGB array
                rgb_array = np.array(rgb_resized)
                rgb_bool = rgb_array > 127  # Boolean mask [H, W, 3]
                
                # Save as NPZ
                base_name = filename.rsplit('_', 1)[0]  # test_00000
                npz_name = f"{base_name}_layer.npz"
                npz_path = os.path.join(output_dir, npz_name)
                np.savez_compressed(npz_path, data=rgb_bool)
                saved_files.append(npz_name)
            
            # Log progress
            if (batch_idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                imgs_done = (batch_idx + 1) * batch_size
                rate = imgs_done / elapsed
                print(f"  Processed {imgs_done}/{len(image_paths)} images ({rate:.1f} img/s)")
    
    elapsed = time.time() - start_time
    print(f"[Segmentation] Completed in {elapsed:.1f}s ({len(image_paths)/elapsed:.1f} img/s)")
    
    return saved_files


def main():
    args = parse_args()
    
    print("=" * 70)
    print("SEGFORMER SEGMENTATION INFERENCE")
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
    
    # Load model
    print(f"\nLoading model from {args.model_path}...")
    model = SegFormerSegmentor.load_pretrained(args.model_path, device=device)
    
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
    effective_batch = args.batch_size * max(1, num_gpus)
    
    saved_files = run_inference(
        model=model,
        image_paths=image_paths,
        device=device,
        output_dir=args.output_dir,
        target_size=args.image_size,
        batch_size=effective_batch,
        num_workers=args.num_workers,
        use_fp16=args.fp16,
        use_tta=args.tta
    )
    
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
