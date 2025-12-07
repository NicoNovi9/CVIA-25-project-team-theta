"""
Fast Multi-GPU Inference script for UNet segmentation model.
Generates NPZ segmentation masks for competition submission.

Submission format:
submission.zip
├── test_00000_layer.npz <-- Segmentation Result
└── ...

Usage:
    python inference_unet.py \
        --model_path model_weights_unet_30epochs/unet_best \
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
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from unet_model import UNetSegmentor


def parse_args():
    parser = argparse.ArgumentParser(description="UNet Segmentation Inference")
    
    # Model path
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to UNet segmentation model directory")
    
    # Data path
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to segmentation test images directory")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="submission_output",
                        help="Directory to save submission files")
    
    # Inference settings
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument("--target_size", type=int, default=512,
                        help="Size to resize images for segmentation")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of data loading workers")
    
    # Flags
    parser.add_argument("--use_tiny", action="store_true",
                        help="Use UNetTiny variant")
    parser.add_argument("--use_small", action="store_true",
                        help="Use UNetSmall variant")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Use FP16 for faster inference")
    
    return parser.parse_args()


# =============================================================================
# DATASET
# =============================================================================

class SegmentationDataset(Dataset):
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


def segmentation_collate_fn(batch):
    """Collate function for segmentation dataset."""
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


def run_segmentation_inference(model, image_paths, device, output_dir, 
                                target_size=512, batch_size=64, num_workers=8, use_fp16=True):
    """
    Run segmentation inference using DataLoader for efficient batching.
    Saves NPZ files directly with RGB binary masks.
    
    UNet outputs class indices:
        0 = background (black)
        1 = spacecraft body (red)
        2 = solar panels (blue)
    
    Output format: RGB boolean mask [H, W, 3] where each channel indicates presence.
    """
    print(f"\n[Segmentation] Running inference on {len(image_paths)} images...")
    print(f"[Segmentation] Batch size: {batch_size}, Workers: {num_workers}")
    
    # Create dataset and dataloader
    dataset = SegmentationDataset(image_paths, target_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=segmentation_collate_fn,
        prefetch_factor=2,
    )
    
    saved_files = []
    start_time = time.time()
    
    # Class to RGB color mapping
    # 0 = background (black), 1 = spacecraft body (red), 2 = solar panels (blue)
    class_to_rgb = {
        0: [0, 0, 0],       # background - black
        1: [255, 0, 0],     # spacecraft body - red
        2: [0, 0, 255],     # solar panels - blue
    }
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Segmentation")):
            images = batch["images"].to(device, non_blocking=True)
            filenames = batch["filenames"]
            orig_ws = batch["orig_ws"]
            orig_hs = batch["orig_hs"]
            
            # Forward pass with FP16
            if use_fp16:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
            else:
                outputs = model(images)
            
            # Get predicted masks (class indices)
            pred_masks = outputs['pred_masks']  # [B, H, W] with values 0, 1, 2
            
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
                
                # Resize RGB mask to original size using nearest neighbor
                rgb_pil = Image.fromarray(rgb_mask, mode='RGB')
                rgb_resized = rgb_pil.resize((orig_w, orig_h), Image.NEAREST)
                
                # Convert to boolean RGB array (threshold at 127)
                rgb_array = np.array(rgb_resized)
                rgb_bool = rgb_array > 127  # Boolean mask [H, W, 3]
                
                # Save as NPZ with 'data' key (matching the expected format)
                base_name = filename.rsplit('_', 1)[0]  # test_00000
                npz_name = f"{base_name}_layer.npz"
                npz_path = os.path.join(output_dir, npz_name)
                np.savez_compressed(npz_path, data=rgb_bool)
                saved_files.append(npz_name)
            
            # Log progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                imgs_done = (batch_idx + 1) * batch_size
                rate = imgs_done / elapsed
                print(f"  [Segmentation] Processed {imgs_done}/{len(image_paths)} images ({rate:.1f} img/s)")
    
    elapsed = time.time() - start_time
    print(f"[Segmentation] Completed in {elapsed:.1f}s ({len(image_paths)/elapsed:.1f} img/s)")
    
    return saved_files


def main():
    args = parse_args()
    
    # =========================================================================
    # SETUP
    # =========================================================================
    print("=" * 70)
    print("UNET SEGMENTATION INFERENCE")
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
    # SEGMENTATION TASK
    # =========================================================================
    print("\n" + "=" * 70)
    print("SPACECRAFT SEGMENTATION")
    print("=" * 70)
    
    # Load segmentation model
    print(f"Loading segmentation model from {args.model_path}...")
    segmentation_model = UNetSegmentor.load_pretrained(
        args.model_path,
        device=device,
        use_tiny_fallback=args.use_tiny,
        use_small_fallback=args.use_small
    )
    
    # Use DataParallel if multiple GPUs
    if num_gpus > 1:
        print(f"Using DataParallel across {num_gpus} GPUs")
        segmentation_model = nn.DataParallel(segmentation_model)
    
    segmentation_model.to(device)
    segmentation_model.eval()
    
    # Get test images
    segmentation_images = get_test_images(args.data_path)
    print(f"Found {len(segmentation_images)} segmentation test images")
    
    # Run segmentation - larger batch for UNet (faster)
    effective_batch = args.batch_size * max(1, num_gpus) * 2  # UNet can handle larger batches
    saved_files = run_segmentation_inference(
        model=segmentation_model,
        image_paths=segmentation_images,
        device=device,
        output_dir=args.output_dir,
        target_size=args.target_size,
        batch_size=effective_batch,
        num_workers=args.num_workers,
        use_fp16=args.fp16
    )
    
    # Free memory
    del segmentation_model
    torch.cuda.empty_cache()
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    total_time = time.time() - total_start
    print("\n" + "=" * 70)
    print("SEGMENTATION INFERENCE COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Output directory: {args.output_dir}")
    print(f"  - NPZ files: {len(saved_files)} files")
    
    # Sample outputs
    print("\nSample segmentation files:")
    for f in sorted(saved_files)[:5]:
        print(f"  {f}")


if __name__ == "__main__":
    main()
