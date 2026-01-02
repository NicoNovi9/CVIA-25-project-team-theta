"""
YOLO Detection Module for SPARK Dataset

This module provides YOLO-based object detection for the SPARK spacecraft dataset.
It includes:
- Dataset conversion from SPARK CSV format to YOLO format
- YOLO model wrapper with DDP support
- Training, evaluation, and inference scripts

Usage:
    # Training
    python train_yolo.py --config config_yolo.yaml
    
    # Evaluation
    python eval_yolo.py --model_path model.pt --data_yaml data.yaml
    
    # Inference
    python inference_yolo.py --model_path model.pt --data_path images/
"""

from .dataset_yolo import (
    CLASS_NAMES,
    CLASS_TO_ID,
    ID_TO_CLASS,
    convert_spark_to_yolo,
    get_augmentation_params,
    verify_yolo_dataset,
)

from .yolo_model import (
    YOLODetector,
    load_yolo_model,
    get_model_info,
    MODEL_SIZES,
)

__all__ = [
    # Dataset
    'CLASS_NAMES',
    'CLASS_TO_ID', 
    'ID_TO_CLASS',
    'convert_spark_to_yolo',
    'get_augmentation_params',
    'verify_yolo_dataset',
    # Model
    'YOLODetector',
    'load_yolo_model',
    'get_model_info',
    'MODEL_SIZES',
]
