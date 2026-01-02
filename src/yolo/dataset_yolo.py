"""
YOLO-compatible dataset converter and loader for SPARK Detection dataset.

This module converts the SPARK dataset from CSV format to YOLO format
and provides utilities for dataset management.
"""

import os
import ast
import shutil
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import yaml


# =============================================================================
# CLASS MAPPINGS (SPARK Dataset - 10 classes)
# =============================================================================
CLASS_NAMES = [
    "VenusExpress",
    "Cheops", 
    "LisaPathfinder",
    "ObservationSat1",
    "Proba2",
    "Proba3",
    "Proba3ocs",
    "Smart1",
    "Soho",
    "XMM Newton"
]

CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}


def convert_bbox_to_yolo(bbox_str, img_width=1024, img_height=1024):
    """
    Convert bounding box from SPARK format to YOLO format.
    
    SPARK format: "(x_min, y_min, x_max, y_max)" or "[x_min, y_min, x_max, y_max]"
    YOLO format: x_center, y_center, width, height (all normalized 0-1)
    
    Args:
        bbox_str: String representation of bounding box
        img_width: Image width (default 1024 for SPARK)
        img_height: Image height (default 1024 for SPARK)
    
    Returns:
        Tuple of (x_center, y_center, width, height) normalized
    """
    # Parse bbox string
    if isinstance(bbox_str, str):
        # Handle both "(x1, y1, x2, y2)" and "[x1, y1, x2, y2]" formats
        bbox_str = bbox_str.strip("()[]")
        x1, y1, x2, y2 = [int(float(x.strip())) for x in bbox_str.split(",")]
    else:
        x1, y1, x2, y2 = bbox_str
    
    # Convert to YOLO format (normalized center + width/height)
    x_center = (x1 + x2) / 2.0 / img_width
    y_center = (y1 + y2) / 2.0 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    # Clamp values to [0, 1]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))
    
    return x_center, y_center, width, height


def process_single_image(row, split, src_image_root, dst_image_dir, dst_label_dir, img_size=1024):
    """
    Process a single image: copy to YOLO directory and create label file.
    
    Args:
        row: DataFrame row with image info
        split: 'train' or 'val'
        src_image_root: Source image root directory
        dst_image_dir: Destination image directory
        dst_label_dir: Destination label directory
        img_size: Image size (assumed square)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        img_name = row["Image name"]
        cls_name = row["Class"]
        bbox_str = row["Bounding box"]
        
        # Source path
        src_path = os.path.join(src_image_root, cls_name, split, img_name)
        
        if not os.path.exists(src_path):
            return False
        
        # Create unique filename to avoid collisions between classes
        new_img_name = f"{cls_name}_{split}_{img_name}"
        dst_img_path = os.path.join(dst_image_dir, new_img_name)
        
        # Copy image
        shutil.copy2(src_path, dst_img_path)
        
        # Convert bbox to YOLO format
        x_center, y_center, width, height = convert_bbox_to_yolo(
            bbox_str, img_width=img_size, img_height=img_size
        )
        
        # Get class ID
        cls_id = CLASS_TO_ID[cls_name]
        
        # Create label file
        label_name = os.path.splitext(new_img_name)[0] + ".txt"
        label_path = os.path.join(dst_label_dir, label_name)
        
        with open(label_path, 'w') as f:
            f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        return True
    
    except Exception as e:
        print(f"Error processing {row.get('Image name', 'unknown')}: {e}")
        return False


def convert_spark_to_yolo(
    data_root: str,
    output_path: str,
    train_csv: str = "train.csv",
    val_csv: str = "val.csv",
    image_root: str = "images",
    img_size: int = 1024,
    num_workers: int = 8,
    force_recreate: bool = False
):
    """
    Convert SPARK dataset to YOLO format.
    
    Args:
        data_root: Root directory of SPARK dataset
        output_path: Output directory for YOLO format dataset
        train_csv: Name of training CSV file
        val_csv: Name of validation CSV file
        image_root: Name of images subdirectory
        img_size: Image size (assumed square)
        num_workers: Number of parallel workers
        force_recreate: If True, recreate even if exists
    
    Returns:
        Path to data.yaml configuration file
    """
    output_path = Path(output_path)
    yaml_path = output_path / "data.yaml"
    
    # Check if already converted
    if yaml_path.exists() and not force_recreate:
        print(f"[Dataset] YOLO dataset already exists at {output_path}")
        print(f"[Dataset] Using existing data.yaml: {yaml_path}")
        return str(yaml_path)
    
    print(f"[Dataset] Converting SPARK dataset to YOLO format...")
    print(f"[Dataset] Source: {data_root}")
    print(f"[Dataset] Destination: {output_path}")
    
    # Create directory structure
    dirs = {
        'train_images': output_path / "images" / "train",
        'val_images': output_path / "images" / "val", 
        'train_labels': output_path / "labels" / "train",
        'val_labels': output_path / "labels" / "val",
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    src_image_root = os.path.join(data_root, image_root)
    
    # Process train and val splits
    for split, csv_name in [('train', train_csv), ('val', val_csv)]:
        csv_path = os.path.join(data_root, csv_name)
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        print(f"[Dataset] Processing {split}: {len(df)} images...")
        
        img_dir = dirs[f'{split}_images']
        lbl_dir = dirs[f'{split}_labels']
        
        # Process images in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for _, row in df.iterrows():
                future = executor.submit(
                    process_single_image,
                    row, split, src_image_root, 
                    str(img_dir), str(lbl_dir), img_size
                )
                futures.append(future)
            
            # Wait for completion with progress bar
            success_count = 0
            for future in tqdm(futures, desc=f"Converting {split}"):
                if future.result():
                    success_count += 1
        
        print(f"[Dataset] {split}: {success_count}/{len(df)} images converted")
    
    # Create data.yaml
    yaml_content = {
        'path': str(output_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: name for i, name in enumerate(CLASS_NAMES)}
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"[Dataset] Created data.yaml: {yaml_path}")
    print(f"[Dataset] Conversion complete!")
    
    return str(yaml_path)


def get_augmentation_params(level: str) -> dict:
    """
    Get augmentation parameters based on level.
    
    Args:
        level: 'none', 'light', 'medium', 'heavy'
    
    Returns:
        Dictionary of augmentation parameters for YOLO training
    """
    params = {
        'none': {
            'fliplr': 0.0,
            'flipud': 0.0,
            'degrees': 0.0,
            'translate': 0.0,
            'scale': 0.0,
            'shear': 0.0,
            'perspective': 0.0,
            'hsv_h': 0.0,
            'hsv_s': 0.0,
            'hsv_v': 0.0,
            'mosaic': 0.0,
            'mixup': 0.0,
            'copy_paste': 0.0,
            'auto_augment': '',
        },
        'light': {
            'fliplr': 0.5,
            'flipud': 0.1,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.1,
            'shear': 0.0,
            'perspective': 0.0,
            'hsv_h': 0.015,
            'hsv_s': 0.5,
            'hsv_v': 0.3,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0,
            'auto_augment': '',
        },
        'medium': {
            'fliplr': 0.5,
            'flipud': 0.2,
            'degrees': 5.0,
            'translate': 0.15,
            'scale': 0.2,
            'shear': 2.0,
            'perspective': 0.0,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'mosaic': 1.0,
            'mixup': 0.1,
            'copy_paste': 0.0,
            'auto_augment': '',
        },
        'heavy': {
            'fliplr': 0.5,
            'flipud': 0.2,
            'degrees': 10.0,
            'translate': 0.15,
            'scale': 0.2,
            'shear': 5.0,
            'perspective': 0.0001,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'mosaic': 1.0,
            'mixup': 0.2,
            'copy_paste': 0.1,
            'auto_augment': 'randaugment',
        },
    }
    
    if level not in params:
        print(f"[Warning] Unknown augmentation level '{level}', using 'light'")
        level = 'light'
    
    return params[level]


def verify_yolo_dataset(dataset_path: str) -> dict:
    """
    Verify YOLO dataset integrity.
    
    Args:
        dataset_path: Path to YOLO dataset
    
    Returns:
        Dictionary with verification results
    """
    dataset_path = Path(dataset_path)
    results = {'valid': True, 'errors': [], 'stats': {}}
    
    for split in ['train', 'val']:
        img_dir = dataset_path / "images" / split
        lbl_dir = dataset_path / "labels" / split
        
        if not img_dir.exists():
            results['valid'] = False
            results['errors'].append(f"Missing {split} images directory")
            continue
        
        if not lbl_dir.exists():
            results['valid'] = False
            results['errors'].append(f"Missing {split} labels directory")
            continue
        
        # Count files
        img_files = list(img_dir.glob("*.[jJpP][pPnN][gG]")) + list(img_dir.glob("*.jpeg"))
        lbl_files = list(lbl_dir.glob("*.txt"))
        
        results['stats'][split] = {
            'images': len(img_files),
            'labels': len(lbl_files),
        }
        
        if len(img_files) != len(lbl_files):
            results['errors'].append(
                f"{split}: image/label count mismatch ({len(img_files)} vs {len(lbl_files)})"
            )
    
    # Check data.yaml
    yaml_path = dataset_path / "data.yaml"
    if not yaml_path.exists():
        results['valid'] = False
        results['errors'].append("Missing data.yaml")
    
    return results


if __name__ == "__main__":
    # Test conversion
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert SPARK dataset to YOLO format")
    parser.add_argument("--data_root", type=str, required=True, help="SPARK dataset root")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers")
    parser.add_argument("--force", action="store_true", help="Force recreation")
    
    args = parser.parse_args()
    
    yaml_path = convert_spark_to_yolo(
        data_root=args.data_root,
        output_path=args.output,
        num_workers=args.workers,
        force_recreate=args.force
    )
    
    # Verify
    results = verify_yolo_dataset(args.output)
    print(f"\n[Verification] Valid: {results['valid']}")
    print(f"[Verification] Stats: {results['stats']}")
    if results['errors']:
        print(f"[Verification] Errors: {results['errors']}")
