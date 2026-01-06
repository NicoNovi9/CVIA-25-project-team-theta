"""
SegFormer Semantic Segmentation Module

This module provides a complete implementation for training, evaluating,
and running inference with SegFormer models for the SPARK segmentation task.

Components:
    - segformer_model.py: SegFormer model wrapper with loss functions
    - train_segformer.py: DeepSpeed distributed training script
    - eval_segformer.py: Evaluation script with metrics
    - inference_segformer.py: Inference script for submission generation
    - config_segformer.yaml: Configuration file for all hyperparameters

Usage:
    from segformer.segformer_model import SegFormerSegmentor
    
    model = SegFormerSegmentor(
        n_classes=3,
        variant='b2',
        pretrained='ade20k'
    )
"""

from .segformer_model import SegFormerSegmentor, list_available_variants

__all__ = ['SegFormerSegmentor', 'list_available_variants']
