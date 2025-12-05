"""
Segmentation dataset for Spark Segmentation task.
3-class segmentation: background (black), spacecraft body (red), solar panels (blue).
Compatible with UNet-style models for DeepSpeed distributed training.

Note: Masks are in JPG format with compression artifacts, so we use distance-based
      classification to map noisy RGB values to the 3 clean classes.
"""

import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random


# Target RGB values for the 3 classes
# 0 = background (black), 1 = spacecraft body (red), 2 = solar panels (blue)
CLASS_COLORS = {
    0: np.array([0, 0, 0]),       # Background - black
    1: np.array([255, 0, 0]),     # Spacecraft body - red
    2: np.array([0, 0, 255]),     # Solar panels - blue
}

CLASS_NAMES = ["background", "spacecraft_body", "solar_panels"]
NUM_CLASSES = 3


def rgb_to_class_mask(rgb_mask):
    """
    Convert RGB mask to class indices, handling JPG compression artifacts.
    
    Uses nearest-neighbor classification based on Euclidean distance in RGB space
    to handle noisy pixel values from JPG compression.
    
    Args:
        rgb_mask: numpy array of shape [H, W, 3] with RGB values
    
    Returns:
        class_mask: numpy array of shape [H, W] with class indices (0, 1, 2)
    """
    H, W, _ = rgb_mask.shape
    
    # Reshape to [H*W, 3] for vectorized distance computation
    pixels = rgb_mask.reshape(-1, 3).astype(np.float32)
    
    # Compute distance to each class color
    distances = np.zeros((pixels.shape[0], NUM_CLASSES), dtype=np.float32)
    for class_idx, color in CLASS_COLORS.items():
        distances[:, class_idx] = np.linalg.norm(pixels - color.astype(np.float32), axis=1)
    
    # Assign each pixel to the nearest class
    class_indices = np.argmin(distances, axis=1)
    
    return class_indices.reshape(H, W).astype(np.int64)


class SparkSegmentationDataset(Dataset):
    """
    Dataset for Spark 3-class Segmentation task.
    
    Classes:
        0 = background (black in mask)
        1 = spacecraft body (red in mask)  
        2 = solar panels (blue in mask)
    
    Expected structure:
        csv_path: CSV file with columns: Class, Image name, Mask name
        image_root/{Class}/{split}/{Image name}
        mask_root/{Class}/{split}/{Mask name}
    """
    def __init__(self, csv_path, image_root, mask_root, split, 
                 target_size=(256, 256), augment=False):
        """
        Args:
            csv_path: Path to CSV file with annotations
            image_root: Root directory for images
            mask_root: Root directory for masks
            split: 'train' or 'val'
            target_size: Tuple (H, W) for resizing images and masks
            augment: Whether to apply data augmentation (for training)
        """
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.mask_root = mask_root
        self.split = split
        self.target_size = target_size
        self.augment = augment and split == 'train'

        # Segmentation classes (fixed for this task)
        self.num_classes = NUM_CLASSES
        self.class_names = CLASS_NAMES
        
        # Satellite classes from CSV (for reference/grouping)
        self.satellite_classes = sorted(self.df["Class"].unique())
        self.satellite_to_idx = {c: i for i, c in enumerate(self.satellite_classes)}
        
        print(f"[SegDataset] Loaded {len(self.df)} samples from {split}")
        print(f"[SegDataset] Segmentation classes: {self.class_names}")
        print(f"[SegDataset] Target size: {target_size}")
        print(f"[SegDataset] Augmentation: {self.augment}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Build paths
        img_path = os.path.join(
            self.image_root, row["Class"], self.split, row["Image name"]
        )
        mask_path = os.path.join(
            self.mask_root, row["Class"], self.split, row["Mask name"]
        )

        # Load image
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Load mask as RGB (to parse the color-coded classes)
        mask = Image.open(mask_path)
        if mask.mode != "RGB":
            mask = mask.convert("RGB")

        # Resize to target size
        if self.target_size is not None:
            img = img.resize((self.target_size[1], self.target_size[0]), Image.BILINEAR)
            mask = mask.resize((self.target_size[1], self.target_size[0]), Image.NEAREST)

        # Apply data augmentation (same transforms for image and mask)
        if self.augment:
            img, mask = self._augment(img, mask)

        # Convert image to tensor and normalize
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Convert RGB mask to class indices (handles JPG artifacts)
        mask_np = np.array(mask)
        class_mask = rgb_to_class_mask(mask_np)
        mask = torch.from_numpy(class_mask).long()

        return {
            "image": img,
            "mask": mask,
            "satellite_class": row["Class"],
            "image_name": row["Image name"]
        }

    def _augment(self, img, mask):
        """Apply random augmentations to image and mask together."""
        # Random horizontal flip
        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        
        # Random vertical flip
        if random.random() > 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask)
        
        # Random rotation (0, 90, 180, 270 degrees)
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            img = TF.rotate(img, angle)
            mask = TF.rotate(mask, angle)
        
        # Random brightness/contrast for image only
        if random.random() > 0.5:
            brightness_factor = random.uniform(0.8, 1.2)
            img = TF.adjust_brightness(img, brightness_factor)
        
        if random.random() > 0.5:
            contrast_factor = random.uniform(0.8, 1.2)
            img = TF.adjust_contrast(img, contrast_factor)
        
        return img, mask


def collate_fn_segmentation(batch):
    """
    Custom collate function for segmentation datasets.
    Stacks images and masks into batches.
    """
    images = torch.stack([item["image"] for item in batch])
    masks = torch.stack([item["mask"] for item in batch])
    satellite_classes = [item["satellite_class"] for item in batch]
    image_names = [item["image_name"] for item in batch]
    
    return {
        "images": images,
        "masks": masks,
        "satellite_classes": satellite_classes,
        "image_names": image_names
    }


if __name__ == "__main__":
    # Quick test of the dataset
    print("Testing SparkSegmentationDataset...")
    
    DATA_ROOT = "/project/scratch/p200981/spark2024"
    
    dataset = SparkSegmentationDataset(
        csv_path=f"{DATA_ROOT}/train.csv",
        image_root=f"{DATA_ROOT}/images",
        mask_root=f"{DATA_ROOT}/mask",
        split="train",
        target_size=(256, 256),
        augment=False
    )
    
    print(f"\nDataset length: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Mask shape: {sample['mask'].shape}")
    print(f"Mask dtype: {sample['mask'].dtype}")
    print(f"Mask unique values: {torch.unique(sample['mask'])}")
    print(f"Satellite class: {sample['satellite_class']}")
    print(f"Image name: {sample['image_name']}")
    
    # Check class distribution in mask
    mask = sample['mask']
    for i, name in enumerate(CLASS_NAMES):
        count = (mask == i).sum().item()
        pct = 100 * count / mask.numel()
        print(f"  Class {i} ({name}): {count} pixels ({pct:.1f}%)")
