"""
DETR-compatible dataset for Spark Detection using CSV-based structure.
"""

import os
import ast
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class SparkDetectionDatasetDETR(Dataset):
    """
    Dataset for Spark Detection task using CSV-based structure.
    Compatible with DETR-style models from HuggingFace.
    
    Expected structure:
        csv_path: CSV file with columns: Class, Image name, Bounding box
        image_root/
        ├── {Class}/
        │   ├── train/
        │   │   └── {Image name}
        │   └── val/
        │       └── {Image name}
    
    Bounding box format in CSV: [x_min, y_min, x_max, y_max] (pixels)
    """
    def __init__(self, csv_path, image_root, split, transform=None, image_processor=None):
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.split = split
        self.transform = transform
        self.image_processor = image_processor

        # Build class mappings from CSV
        class_names = sorted(self.df["Class"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.classes = class_names
        self.id2label = {i: name for i, name in enumerate(class_names)}
        self.label2id = {name: i for i, name in enumerate(class_names)}
        self.num_classes = len(class_names)
        
        print(f"[Dataset] Loaded {len(self.df)} images from {split}")
        print(f"[Dataset] Classes ({self.num_classes}): {self.id2label}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(
            self.image_root,
            row["Class"],
            self.split,
            row["Image name"]
        )

        # 1) Load PIL image
        img = Image.open(img_path)

        # 2) Force RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        W, H = img.size

        # 3) Load annotation from CSV
        bbox = ast.literal_eval(row["Bounding box"])  # [x_min, y_min, x_max, y_max]
        label = self.class_to_idx[row["Class"]]
        
        xmin, ymin, xmax, ymax = bbox
        
        # Clamp to image bounds
        xmin = max(0, min(W - 1, xmin))
        ymin = max(0, min(H - 1, ymin))
        xmax = max(1, min(W, xmax))
        ymax = max(1, min(H, ymax))
        
        boxes = [[xmin, ymin, xmax, ymax]]
        labels = [label]
        areas = [(xmax - xmin) * (ymax - ymin)]

        # 4) Apply image processor (for DETR)
        if self.image_processor is not None:
            img_np = np.array(img)
            
            # Create COCO-style annotations for DETR processor
            # IMPORTANT: bbox format is [x_min, y_min, width, height] for COCO
            coco_annotations = []
            for i, (box, label, area) in enumerate(zip(boxes, labels, areas)):
                xmin, ymin, xmax, ymax = box
                coco_annotations.append({
                    "image_id": idx,
                    "category_id": label,  # 0-indexed class ID from YOLO
                    "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],  # COCO format: x, y, w, h
                    "area": area,
                    "iscrowd": 0,
                })
            
            target = {
                "image_id": idx,
                "annotations": coco_annotations
            }
            
            # DETR image processor handles:
            # - Resizing images
            # - Normalizing pixel values  
            # - Converting boxes to normalized cxcywh format
            encoding = self.image_processor(
                images=img_np,
                annotations=[target],
                return_tensors="pt"
            )
            
            # Remove batch dimension
            pixel_values = encoding["pixel_values"].squeeze(0)
            detr_labels = encoding["labels"][0]
            
            return {"pixel_values": pixel_values, "labels": detr_labels}
        
        else:
            # Fallback without processor
            if self.transform is not None:
                img = self.transform(img)
            else:
                img = T.ToTensor()(img)

            # Return normalized cxcywh format for DETR
            boxes_normalized = []
            for box in boxes:
                xmin, ymin, xmax, ymax = box
                cx = ((xmin + xmax) / 2) / W
                cy = ((ymin + ymax) / 2) / H
                bw = (xmax - xmin) / W
                bh = (ymax - ymin) / H
                boxes_normalized.append([cx, cy, bw, bh])
            
            target = {
                "boxes": torch.tensor(boxes_normalized, dtype=torch.float32),
                "class_labels": torch.tensor(labels, dtype=torch.int64),
                "image_id": torch.tensor([idx]),
                "orig_size": torch.tensor([H, W]),
                "area": torch.tensor(areas, dtype=torch.float32),
                "iscrowd": torch.zeros(len(labels), dtype=torch.int64),
            }

            return {"pixel_values": img, "labels": target}


def collate_fn_detr(batch):
    """
    Custom collate function for DETR that handles variable-sized labels.
    """
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = [item["labels"] for item in batch]
    return {"pixel_values": pixel_values, "labels": labels}
