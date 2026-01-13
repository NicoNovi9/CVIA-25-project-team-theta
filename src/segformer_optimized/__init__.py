"""
SegFormer Optimized - YOLO-guided Semantic Segmentation

This package provides an optimized version of SegFormer that uses YOLO detection
to focus segmentation on spacecraft regions. Key features:

- YOLO-guided bounding box extraction
- Resolution preservation for small objects (padding instead of downsampling)
- Efficient computation (segmentation only on ROI)
- Full pipeline integration (training, evaluation, inference)

Main components:
- SegFormerOptimizedSegmentor: Model with YOLO-guided segmentation
- BBoxProcessor: Handles bbox extraction and inverse mapping
- SparkSegmentationOptimizedDataset: Dataset with YOLO integration
- YOLODetector: YOLO wrapper with caching support
"""

from .segformer_optimized_model import (
    SegFormerOptimizedSegmentor,
    BBoxProcessor,
    MultiClassSegmentationLoss,
    FocalLoss,
    list_available_variants
)

from .dataset_segformer_optimized import (
    SparkSegmentationOptimizedDataset,
    YOLODetector,
    collate_fn_segmentation_optimized,
    CLASS_COLORS,
    CLASS_NAMES,
    NUM_CLASSES
)

__version__ = "1.0.0"

__all__ = [
    # Model
    'SegFormerOptimizedSegmentor',
    'BBoxProcessor',
    'MultiClassSegmentationLoss',
    'FocalLoss',
    'list_available_variants',
    
    # Dataset
    'SparkSegmentationOptimizedDataset',
    'YOLODetector',
    'collate_fn_segmentation_optimized',
    'CLASS_COLORS',
    'CLASS_NAMES',
    'NUM_CLASSES',
]
