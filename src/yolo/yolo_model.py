"""
YOLO Model wrapper for object detection using Ultralytics.

This module provides a wrapper around Ultralytics YOLO for training on the SPARK dataset.
The detection head is randomly initialized (no COCO pretrained weights for the head),
while the backbone can use pretrained weights.

Compatible with DDP distributed training via Ultralytics built-in support.
"""

import os
import torch
import torch.nn as nn
from pathlib import Path
import yaml

from dataset_yolo import CLASS_NAMES, ID_TO_CLASS, CLASS_TO_ID


# Available YOLO model sizes
MODEL_SIZES = {
    'n': 'yolo11n.pt',   # Nano - fastest, smallest
    's': 'yolo11s.pt',   # Small
    'm': 'yolo11m.pt',   # Medium
    'l': 'yolo11l.pt',   # Large
    'x': 'yolo11x.pt',   # XLarge - most accurate, slowest
}


def get_model_path(size: str, base: str = "yolo11") -> str:
    """
    Get the model filename for a given size.
    
    Args:
        size: Model size ('n', 's', 'm', 'l', 'x')
        base: Base model name (default: 'yolo11')
    
    Returns:
        Model filename (e.g., 'yolo11n.pt')
    """
    if size not in MODEL_SIZES:
        raise ValueError(f"Invalid model size '{size}'. Choose from: {list(MODEL_SIZES.keys())}")
    
    return f"{base}{size}.pt"


def load_yolo_model(
    model_size: str = 'n',
    model_base: str = 'yolo11',
    num_classes: int = 10,
    pretrained_backbone: bool = True
):
    """
    Load YOLO model with custom number of classes.
    
    The detection head is NOT pretrained on COCO - it's randomly initialized
    for the SPARK dataset classes. Only the backbone uses pretrained weights.
    
    Args:
        model_size: Size variant ('n', 's', 'm', 'l', 'x')
        model_base: Base model (default: 'yolo11')
        num_classes: Number of classes (10 for SPARK)
        pretrained_backbone: Whether to use pretrained backbone weights
    
    Returns:
        YOLO model instance from Ultralytics
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics package not found. Install with: pip install ultralytics"
        )
    
    model_name = get_model_path(model_size, model_base)
    
    print(f"[YOLOModel] Loading {model_name}...")
    print(f"[YOLOModel] Number of classes: {num_classes}")
    print(f"[YOLOModel] Pretrained backbone: {pretrained_backbone}")
    
    # Load the pretrained model
    # Ultralytics YOLO will automatically adjust the head for new number of classes
    # when training with a different nc (number of classes)
    model = YOLO(model_name)
    
    return model


class YOLODetector:
    """
    Wrapper class for YOLO model providing a unified interface.
    
    This class handles:
    - Model loading with custom classes
    - Training with DDP support
    - Validation and inference
    - Model saving/loading
    """
    
    def __init__(
        self,
        model_size: str = 'n',
        model_base: str = 'yolo11',
        num_classes: int = 10,
        class_names: list = None,
        pretrained_backbone: bool = True
    ):
        """
        Initialize YOLO detector.
        
        Args:
            model_size: Model size ('n', 's', 'm', 'l', 'x')
            model_base: Base model name
            num_classes: Number of classes
            class_names: List of class names
            pretrained_backbone: Use pretrained backbone
        """
        self.model_size = model_size
        self.model_base = model_base
        self.num_classes = num_classes
        self.class_names = class_names or CLASS_NAMES
        self.pretrained_backbone = pretrained_backbone
        
        # Class mappings
        self.id2label = {i: name for i, name in enumerate(self.class_names)}
        self.label2id = {name: i for i, name in enumerate(self.class_names)}
        
        # Load model
        self.model = load_yolo_model(
            model_size=model_size,
            model_base=model_base,
            num_classes=num_classes,
            pretrained_backbone=pretrained_backbone
        )
        
        print(f"[YOLODetector] Initialized with {num_classes} classes")
        print(f"[YOLODetector] Classes: {self.class_names}")
    
    def train(
        self,
        data_yaml: str,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        patience: int = 15,
        workers: int = 8,
        device: str = None,
        project: str = "yolo_spark",
        name: str = "",
        amp: bool = True,
        resume: bool = False,
        save_period: int = 10,
        verbose: bool = True,
        **augmentation_params
    ):
        """
        Train the YOLO model.
        
        Args:
            data_yaml: Path to data.yaml configuration
            epochs: Number of training epochs
            imgsz: Input image size
            batch: Batch size per GPU
            patience: Early stopping patience
            workers: Number of data loading workers
            device: Device(s) to use (None for auto)
            project: Project directory
            name: Run name
            amp: Use mixed precision
            resume: Resume training from checkpoint
            save_period: Save checkpoint every N epochs
            verbose: Verbose output
            **augmentation_params: Augmentation parameters
        
        Returns:
            Training results
        """
        print(f"\n[YOLODetector] Starting training...")
        print(f"[YOLODetector] Data: {data_yaml}")
        print(f"[YOLODetector] Epochs: {epochs}, Image size: {imgsz}")
        print(f"[YOLODetector] Batch size: {batch}, Patience: {patience}")
        
        # Build training arguments
        train_args = {
            'data': data_yaml,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch,
            'patience': patience,
            'workers': workers,
            'project': project,
            'name': name if name else f"{self.model_base}{self.model_size}_{imgsz}",
            'amp': amp,
            'verbose': verbose,
            'save_period': save_period,
            'exist_ok': True,  # Overwrite existing run
            'pretrained': self.pretrained_backbone,
        }
        
        # Add device if specified
        if device is not None:
            train_args['device'] = device
        
        # Add resume if specified
        if resume:
            train_args['resume'] = resume
        
        # Add augmentation parameters
        valid_aug_params = [
            'fliplr', 'flipud', 'degrees', 'translate', 'scale', 'shear',
            'perspective', 'hsv_h', 'hsv_s', 'hsv_v', 'mosaic', 'mixup',
            'copy_paste', 'auto_augment'
        ]
        
        for param in valid_aug_params:
            if param in augmentation_params:
                value = augmentation_params[param]
                # Skip empty strings for auto_augment
                if param == 'auto_augment' and value == '':
                    continue
                train_args[param] = value
        
        # Train
        results = self.model.train(**train_args)
        
        return results
    
    def val(
        self,
        data_yaml: str = None,
        imgsz: int = 640,
        batch: int = 16,
        conf: float = 0.001,
        iou: float = 0.6,
        device: str = None,
        verbose: bool = True
    ):
        """
        Validate the model.
        
        Args:
            data_yaml: Path to data.yaml (uses training data if None)
            imgsz: Input image size
            batch: Batch size
            conf: Confidence threshold
            iou: IoU threshold for NMS
            device: Device to use
            verbose: Verbose output
        
        Returns:
            Validation metrics
        """
        val_args = {
            'imgsz': imgsz,
            'batch': batch,
            'conf': conf,
            'iou': iou,
            'verbose': verbose,
        }
        
        if data_yaml:
            val_args['data'] = data_yaml
        
        if device:
            val_args['device'] = device
        
        results = self.model.val(**val_args)
        
        return results
    
    def predict(
        self,
        source,
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        device: str = None,
        save: bool = False,
        save_txt: bool = False,
        save_conf: bool = False,
        verbose: bool = False
    ):
        """
        Run inference on images.
        
        Args:
            source: Image source (path, list of paths, numpy array, etc.)
            imgsz: Input image size
            conf: Confidence threshold
            iou: IoU threshold for NMS
            device: Device to use
            save: Save annotated images
            save_txt: Save results as text files
            save_conf: Save confidence scores
            verbose: Verbose output
        
        Returns:
            List of Results objects
        """
        predict_args = {
            'source': source,
            'imgsz': imgsz,
            'conf': conf,
            'iou': iou,
            'save': save,
            'save_txt': save_txt,
            'save_conf': save_conf,
            'verbose': verbose,
        }
        
        if device:
            predict_args['device'] = device
        
        results = self.model.predict(**predict_args)
        
        return results
    
    def export(self, format: str = 'onnx', **kwargs):
        """
        Export model to different format.
        
        Args:
            format: Export format ('onnx', 'torchscript', etc.)
            **kwargs: Additional export arguments
        
        Returns:
            Path to exported model
        """
        return self.model.export(format=format, **kwargs)
    
    def save(self, save_path: str):
        """
        Save model weights.
        
        Args:
            save_path: Path to save model
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        model_path = save_path / "best.pt"
        if hasattr(self.model, 'ckpt'):
            torch.save(self.model.ckpt, model_path)
        
        # Save configuration
        config = {
            'model_size': self.model_size,
            'model_base': self.model_base,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'id2label': self.id2label,
            'label2id': self.label2id,
        }
        
        config_path = save_path / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"[YOLODetector] Model saved to {save_path}")
    
    @classmethod
    def from_pretrained(cls, model_path: str, device: str = None):
        """
        Load model from saved checkpoint.
        
        Args:
            model_path: Path to saved model directory or .pt file
            device: Device to load model on
        
        Returns:
            YOLODetector instance
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics package not found. Install with: pip install ultralytics"
            )
        
        model_path = Path(model_path)
        
        # Find model file
        if model_path.suffix == '.pt':
            pt_path = model_path
            config_path = model_path.parent / "config.yaml"
        else:
            # Look for best.pt in directory
            pt_path = model_path / "best.pt"
            if not pt_path.exists():
                pt_path = model_path / "weights" / "best.pt"
            config_path = model_path / "config.yaml"
        
        if not pt_path.exists():
            raise FileNotFoundError(f"Model weights not found at {pt_path}")
        
        print(f"[YOLODetector] Loading from {pt_path}")
        
        # Load configuration if available
        config = {}
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        
        # Create instance
        instance = cls.__new__(cls)
        instance.model_size = config.get('model_size', 'n')
        instance.model_base = config.get('model_base', 'yolo11')
        instance.num_classes = config.get('num_classes', 10)
        instance.class_names = config.get('class_names', CLASS_NAMES)
        instance.id2label = config.get('id2label', {i: name for i, name in enumerate(CLASS_NAMES)})
        instance.label2id = config.get('label2id', {name: i for i, name in enumerate(CLASS_NAMES)})
        instance.pretrained_backbone = config.get('pretrained_backbone', True)
        
        # Load model
        instance.model = YOLO(str(pt_path))
        
        print(f"[YOLODetector] Loaded model with {instance.num_classes} classes")
        
        return instance
    
    def __repr__(self):
        return (
            f"YOLODetector(model={self.model_base}{self.model_size}, "
            f"num_classes={self.num_classes})"
        )


def get_model_info(model_size: str = 'n') -> dict:
    """
    Get information about YOLO model variants.
    
    Args:
        model_size: Model size ('n', 's', 'm', 'l', 'x')
    
    Returns:
        Dictionary with model information
    """
    info = {
        'n': {
            'name': 'YOLOv11 Nano',
            'params': '~2.6M',
            'flops': '~6.5G',
            'speed': 'Fastest',
            'accuracy': 'Good',
        },
        's': {
            'name': 'YOLOv11 Small', 
            'params': '~9.4M',
            'flops': '~21.5G',
            'speed': 'Fast',
            'accuracy': 'Better',
        },
        'm': {
            'name': 'YOLOv11 Medium',
            'params': '~20.1M',
            'flops': '~68.0G',
            'speed': 'Medium',
            'accuracy': 'Great',
        },
        'l': {
            'name': 'YOLOv11 Large',
            'params': '~25.3M',
            'flops': '~86.9G',
            'speed': 'Slower',
            'accuracy': 'Excellent',
        },
        'x': {
            'name': 'YOLOv11 XLarge',
            'params': '~56.9M',
            'flops': '~194.9G',
            'speed': 'Slowest',
            'accuracy': 'Best',
        },
    }
    
    if model_size not in info:
        raise ValueError(f"Invalid model size '{model_size}'")
    
    return info[model_size]


if __name__ == "__main__":
    # Test model loading
    print("Testing YOLO model loading...")
    
    detector = YOLODetector(
        model_size='n',
        num_classes=10,
        class_names=CLASS_NAMES
    )
    
    print(f"\nModel: {detector}")
    print(f"Classes: {detector.class_names}")
    
    # Print model info
    for size in ['n', 's', 'm', 'l', 'x']:
        info = get_model_info(size)
        print(f"\n{info['name']}:")
        print(f"  Params: {info['params']}")
        print(f"  Speed: {info['speed']}")
        print(f"  Accuracy: {info['accuracy']}")
