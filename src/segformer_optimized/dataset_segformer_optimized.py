"""
Dataset for SegFormer Optimized with precomputed bounding box extraction.

This dataset:
1. Reads precomputed spacecraft bounding boxes from CSV
2. Extracts the bounding box region from image and mask
3. Pads (if smaller) or downsamples (if larger) to 512x512
4. Returns the prepared crops for SegFormer training

The key insight is that most spacecraft bounding boxes are smaller than 512x512,
so padding preserves full resolution. Only larger boxes need downsampling.

Bounding boxes are precomputed using a YOLO detection model and saved to CSV.
This separates detection from segmentation training, avoiding distributed
training issues with on-the-fly YOLO inference.
"""

import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import random
from typing import Optional, Dict, Any, Tuple, List


# Target RGB values for the 3 classes
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
    """
    H, W, _ = rgb_mask.shape
    pixels = rgb_mask.reshape(-1, 3).astype(np.float32)
    
    distances = np.zeros((pixels.shape[0], NUM_CLASSES), dtype=np.float32)
    for class_idx, color in CLASS_COLORS.items():
        distances[:, class_idx] = np.linalg.norm(pixels - color.astype(np.float32), axis=1)
    
    class_indices = np.argmin(distances, axis=1)
    return class_indices.reshape(H, W).astype(np.int64)


def parse_bbox_string(bbox_str):
    """
    Parse bounding box string from CSV.
    
    Args:
        bbox_str: String in format "(x_min, y_min, x_max, y_max)" or empty string/NaN
    
    Returns:
        Tuple (x_min, y_min, x_max, y_max) or None if no bbox
    """
    # Handle NaN values from pandas (they come as float)
    if pd.isna(bbox_str):
        return None
    
    # Handle non-string types
    if not isinstance(bbox_str, str):
        return None
    
    # Handle empty strings
    if not bbox_str or bbox_str.strip() == "":
        return None
    
    try:
        # Remove parentheses and split by comma
        bbox_str = bbox_str.strip().strip("()").strip()
        parts = [int(x.strip()) for x in bbox_str.split(",")]
        
        if len(parts) == 4:
            return tuple(parts)
        else:
            return None
    except:
        return None


class BBoxDetectionLoader:
    """
    Loader for precomputed bounding box detections from CSV.
    
    This replaces the YOLODetector to avoid running detection during training,
    which can cause distributed training issues.
    
    Matching is done by line number: the i-th line in the ground truth CSV
    corresponds to the i-th line in the detection CSV. This ensures correct
    matching even when multiple images have the same filename but different classes.
    """
    
    def __init__(self, detection_csv_path: str):
        """
        Args:
            detection_csv_path: Path to CSV file with precomputed detections
                Format: Image name,Class,Bounding box
                Example: image001.jpg,VenusExpress,"(x_min, y_min, x_max, y_max)"
        """
        self.detection_csv_path = detection_csv_path
        self._detections_by_line = []  # Store bboxes by line number
        
        self._load_detections()
    
    def _load_detections(self):
        """Load all detections from CSV into memory, indexed by line number."""
        if not os.path.exists(self.detection_csv_path):
            raise FileNotFoundError(f"Detection CSV not found: {self.detection_csv_path}")
        
        print(f"[BBoxLoader] Loading detections from {self.detection_csv_path}")
        
        df = pd.read_csv(self.detection_csv_path)
        
        # Expected columns: Image name, Class, Bounding box
        if 'Image name' not in df.columns or 'Bounding box' not in df.columns:
            raise ValueError(f"Detection CSV must have 'Image name' and 'Bounding box' columns")
        
        # Parse all bounding boxes and store by line number
        for idx, row in df.iterrows():
            bbox_str = row['Bounding box']
            bbox = parse_bbox_string(bbox_str)
            self._detections_by_line.append(bbox)
        
        # Statistics
        detected = sum(1 for bbox in self._detections_by_line if bbox is not None)
        total = len(self._detections_by_line)
        
        print(f"[BBoxLoader] Loaded {total} detections (by line number)")
        print(f"[BBoxLoader] With bbox: {detected}/{total} ({100*detected/total:.1f}%)")
        print(f"[BBoxLoader] Without bbox: {total-detected}/{total} ({100*(total-detected)/total:.1f}%)")
    
    def get_bbox_by_line(self, line_idx: int) -> Optional[Tuple[int, int, int, int]]:
        """
        Get bounding box by line number in the CSV.
        
        Args:
            line_idx: Line index (0-based) in the detection CSV
        
        Returns:
            Tuple (x_min, y_min, x_max, y_max) or None if no detection or out of range
        """
        if 0 <= line_idx < len(self._detections_by_line):
            return self._detections_by_line[line_idx]
        return None
    
    def __len__(self):
        """Number of loaded detections."""
        return len(self._detections_by_line)


class SparkSegmentationOptimizedDataset(Dataset):
    """
    Dataset for SegFormer Optimized with precomputed bounding box extraction.
    
    Features:
    - Reads precomputed spacecraft bounding boxes from CSV
    - Extracts and pads/resizes bbox region to target_size (512x512)
    - Most bbox are smaller than 512, so they get padded (preserving resolution)
    - Larger bbox are downsampled
    
    Classes:
        0 = background (black in mask)
        1 = spacecraft body (red in mask)  
        2 = solar panels (blue in mask)
    """
    
    def __init__(
        self, 
        csv_path: str,
        image_root: str,
        mask_root: str,
        split: str,
        detection_csv: str,
        target_size: int = 512,
        bbox_expansion: float = 1.1,
        fallback_mode: str = 'full_image',
        augment: bool = False
    ):
        """
        Args:
            csv_path: Path to CSV file with annotations
            image_root: Root directory for images
            mask_root: Root directory for masks
            split: 'train' or 'val'
            detection_csv: Path to CSV with precomputed bounding boxes
            target_size: Target size for segmentation (default 512)
            bbox_expansion: Factor to expand bounding boxes (default 1.1)
            fallback_mode: How to handle missing detections ('full_image' or 'center_crop')
            augment: Whether to apply data augmentation
        """
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.mask_root = mask_root
        self.split = split
        self.target_size = target_size
        self.augment = augment and split == 'train'
        self.bbox_expansion = bbox_expansion
        self.fallback_mode = fallback_mode
        
        self.num_classes = NUM_CLASSES
        self.class_names = CLASS_NAMES
        
        # Satellite classes from CSV
        self.satellite_classes = sorted(self.df["Class"].unique())
        
        # Load precomputed detections
        self.bbox_loader = BBoxDetectionLoader(detection_csv)
        
        print(f"[SegOptDataset] Loaded {len(self.df)} samples from {split}")
        print(f"[SegOptDataset] Target size: {target_size}")
        print(f"[SegOptDataset] BBox expansion: {self.bbox_expansion}")
        print(f"[SegOptDataset] Fallback mode: {self.fallback_mode}")
        print(f"[SegOptDataset] Augmentation: {self.augment}")
    
    def __len__(self):
        return len(self.df)
    
    def _expand_bbox(self, bbox: Tuple[int, int, int, int], 
                     img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """Expand bounding box by the expansion factor."""
        x_min, y_min, x_max, y_max = bbox
        
        width = x_max - x_min
        height = y_max - y_min
        
        expand_w = int(width * (self.bbox_expansion - 1) / 2)
        expand_h = int(height * (self.bbox_expansion - 1) / 2)
        
        x_min = max(0, x_min - expand_w)
        y_min = max(0, y_min - expand_h)
        x_max = min(img_width, x_max + expand_w)
        y_max = min(img_height, y_max + expand_h)
        
        return (x_min, y_min, x_max, y_max)
    
    def _extract_and_prepare(self, image: torch.Tensor, bbox: Tuple[int, int, int, int],
                            mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Extract bbox region and prepare for SegFormer.
        
        Key optimization to prevent warping:
        - If bbox fits in 512x512: pad directly
        - If bbox exceeds 512x512: first pad to SQUARE (max dimension), then downsample
        
        Returns:
            - crop_image: [C, target_size, target_size]
            - crop_mask: [target_size, target_size]
            - metadata: Dictionary with extraction info
        """
        x_min, y_min, x_max, y_max = bbox
        _, H, W = image.shape
        
        # Clamp coordinates
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(W, x_max)
        y_max = min(H, y_max)
        
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        
        # Extract crop
        crop_image = image[:, y_min:y_max, x_min:x_max]
        crop_mask = mask[y_min:y_max, x_min:x_max]
        
        # Determine if padding or downsampling
        needs_downsampling = bbox_width > self.target_size or bbox_height > self.target_size
        
        if not needs_downsampling:
            # Small enough - pad directly to target size
            pad_right = self.target_size - bbox_width
            pad_bottom = self.target_size - bbox_height
            
            # Pad image with zeros (black)
            prepared_image = F.pad(crop_image, (0, pad_right, 0, pad_bottom), value=0)
            
            # Pad mask with background (0)
            prepared_mask = F.pad(crop_mask.unsqueeze(0), (0, pad_right, 0, pad_bottom), value=0).squeeze(0)
            
            scale_factor = 1.0
            padded_to_square = False
            square_size = None
        else:
            # Too large - pad to SQUARE first, then downsample to prevent warping
            max_dim = max(bbox_width, bbox_height)
            
            # Pad to square
            pad_right = max_dim - bbox_width
            pad_bottom = max_dim - bbox_height
            
            # Pad image with zeros (black)
            square_image = F.pad(crop_image, (0, pad_right, 0, pad_bottom), value=0)
            
            # Pad mask with background (0)
            square_mask = F.pad(crop_mask.unsqueeze(0), (0, pad_right, 0, pad_bottom), value=0).squeeze(0)
            
            # Now downsample the square to target size (no warping!)
            prepared_image = F.interpolate(
                square_image.unsqueeze(0),
                size=(self.target_size, self.target_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            
            prepared_mask = F.interpolate(
                square_mask.unsqueeze(0).unsqueeze(0).float(),
                size=(self.target_size, self.target_size),
                mode='nearest'
            ).squeeze(0).squeeze(0).long()
            
            scale_factor = self.target_size / max_dim
            padded_to_square = True
            square_size = max_dim
        
        metadata = {
            'original_bbox': (x_min, y_min, x_max, y_max),
            'bbox_width': bbox_width,
            'bbox_height': bbox_height,
            'needs_downsampling': needs_downsampling,
            'padded_to_square': padded_to_square,
            'square_size': square_size,
            'scale_factor': scale_factor,
            'original_image_size': (H, W)
        }
        
        return prepared_image, prepared_mask, metadata
    
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
        orig_width, orig_height = img.size
        
        # Load mask
        mask = Image.open(mask_path)
        if mask.mode != "RGB":
            mask = mask.convert("RGB")
        
        # Get precomputed bounding box by line number
        # The idx in the ground truth CSV corresponds to the same line in detection CSV
        bbox = self.bbox_loader.get_bbox_by_line(idx)
        
        # Handle no detection
        if bbox is None:
            if self.fallback_mode == 'full_image':
                bbox = (0, 0, orig_width, orig_height)
            else:  # center_crop
                center_x, center_y = orig_width // 2, orig_height // 2
                crop_size = min(orig_width, orig_height, self.target_size)
                half_size = crop_size // 2
                bbox = (center_x - half_size, center_y - half_size, 
                       center_x + half_size, center_y + half_size)
        
        # Expand bbox
        bbox = self._expand_bbox(bbox, orig_width, orig_height)
        
        # Convert to tensor
        img_tensor = TF.to_tensor(img)  # [C, H, W]
        
        # Convert mask to class indices
        mask_np = np.array(mask)
        class_mask = rgb_to_class_mask(mask_np)
        mask_tensor = torch.from_numpy(class_mask).long()
        
        # Extract and prepare bbox region
        crop_image, crop_mask, metadata = self._extract_and_prepare(
            img_tensor, bbox, mask_tensor
        )
        
        # Apply augmentation (to the crop)
        if self.augment:
            crop_image, crop_mask = self._augment(crop_image, crop_mask)
        
        # Normalize image
        crop_image = TF.normalize(crop_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        return {
            "image": crop_image,  # [C, target_size, target_size]
            "mask": crop_mask,    # [target_size, target_size]
            "satellite_class": row["Class"],
            "image_name": row["Image name"],
            "bbox": bbox,
            "metadata": metadata
        }
    
    def _augment(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply augmentations to crop (image and mask together)."""
        # Random horizontal flip
        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask.unsqueeze(0)).squeeze(0)
        
        # Random vertical flip
        if random.random() > 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask.unsqueeze(0)).squeeze(0)
        
        # Random rotation (90, 180, 270)
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            img = TF.rotate(img, angle)
            mask = TF.rotate(mask.unsqueeze(0), angle).squeeze(0)
        
        # Color augmentation (image only)
        if random.random() > 0.5:
            brightness_factor = random.uniform(0.8, 1.2)
            img = TF.adjust_brightness(img, brightness_factor)
        
        if random.random() > 0.5:
            contrast_factor = random.uniform(0.8, 1.2)
            img = TF.adjust_contrast(img, contrast_factor)
        
        return img, mask


def collate_fn_segmentation_optimized(batch):
    """
    Custom collate function for optimized segmentation dataset.
    """
    images = torch.stack([item["image"] for item in batch])
    masks = torch.stack([item["mask"] for item in batch])
    satellite_classes = [item["satellite_class"] for item in batch]
    image_names = [item["image_name"] for item in batch]
    bboxes = [item["bbox"] for item in batch]
    metadatas = [item["metadata"] for item in batch]
    
    return {
        "images": images,
        "masks": masks,
        "satellite_classes": satellite_classes,
        "image_names": image_names,
        "bboxes": bboxes,
        "metadatas": metadatas
    }


if __name__ == "__main__":
    # Quick test
    print("Testing SparkSegmentationOptimizedDataset...")
    
    dataset = SparkSegmentationOptimizedDataset(
        csv_path="/project/scratch/p200981/spark2024/val.csv",
        image_root="/project/scratch/p200981/spark2024/images",
        mask_root="/project/scratch/p200981/spark2024/mask",
        split="val",
        detection_csv="/project/scratch/p200981/spark2024/val_detection.csv",
        target_size=512,
        bbox_expansion=1.1,
        fallback_mode='full_image',
        augment=False
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    print(f"\nSample:")
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Mask shape: {sample['mask'].shape}")
    print(f"  BBox: {sample['bbox']}")
    print(f"  Needs padding: {sample['metadata']['needs_padding']}")
    print(f"  Original bbox size: {sample['metadata']['bbox_width']}x{sample['metadata']['bbox_height']}")
    
    print("\n✅ Dataset test passed!")
