"""
DETR Model wrapper for object detection using HuggingFace Transformers.
Compatible with DeepSpeed distributed training.
"""

import torch
import torch.nn as nn
from transformers import DetrImageProcessor, DetrForObjectDetection

from transformers import logging as hf_logging
hf_logging.set_verbosity_error()  # suppress unused-weights warnings


# Model configuration options
MODEL_NAME = "facebook/detr-resnet-50"  # Pretrained DETR with ResNet-50 backbone
# Alternative models:
# - "facebook/detr-resnet-101" (larger backbone)
# - "microsoft/conditional-detr-resnet-50" (faster convergence)


def load_detr_model(model_name, num_classes, id2label, label2id):
    """
    Load DETR model with pretrained weights and adapt for custom number of classes.
    
    Args:
        model_name: HuggingFace model identifier
        num_classes: Number of classes in your dataset
        id2label: Dictionary mapping class ids to class names
        label2id: Dictionary mapping class names to class ids
    
    Returns:
        model: DETR model ready for fine-tuning
        image_processor: Image processor for preprocessing
    """
    # Load image processor without resizing
    image_processor = DetrImageProcessor.from_pretrained(
        model_name,
        do_resize=False,
    )
    
    # Load model with custom number of classes
    # ignore_mismatched_sizes=True allows loading pretrained weights 
    # even though classification head size differs
    model = DetrForObjectDetection.from_pretrained(
        model_name,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    
    return model, image_processor


class DETRDetector(nn.Module):
    """
    Wrapper around HuggingFace DETR that provides a simpler interface
    for training compatible with the existing DeepSpeed training loop.
    
    Unlike YOLO wrapper which returns (logits, bbox), DETR computes its
    own loss internally when labels are provided.
    """
    def __init__(self, num_classes, model_name="facebook/detr-resnet-50", 
                 id2label=None, label2id=None):
        super().__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Create default label mappings if not provided
        if id2label is None:
            id2label = {i: f"class_{i}" for i in range(num_classes)}
        if label2id is None:
            label2id = {v: k for k, v in id2label.items()}
        
        self.id2label = id2label
        self.label2id = label2id
        
        # Load the DETR model and processor
        self.model, self.image_processor = load_detr_model(
            model_name=model_name,
            num_classes=num_classes,
            id2label=id2label,
            label2id=label2id
        )
        
        print(f"[DETRDetector] Loaded {model_name}")
        print(f"[DETRDetector] Classes: {num_classes}")
    
    def forward(self, pixel_values, labels=None):
        """
        Forward pass through DETR.
        
        Args:
            pixel_values: Preprocessed images [B, C, H, W]
            labels: Optional list of dicts with 'class_labels' and 'boxes'
                    When provided, DETR computes and returns loss
        
        Returns:
            outputs: DETR outputs containing:
                - loss (if labels provided)
                - logits: [B, num_queries, num_classes+1]
                - pred_boxes: [B, num_queries, 4] in cxcywh format
        """
        outputs = self.model(pixel_values=pixel_values, labels=labels)
        return outputs
    
    def get_param_groups(self, backbone_lr, default_lr):
        """
        Create parameter groups with different learning rates.
        Backbone gets lower LR since it's pretrained.
        
        Args:
            backbone_lr: Learning rate for backbone (ResNet)
            default_lr: Learning rate for transformer/heads
            
        Returns:
            List of param group dicts for optimizer
        """
        backbone_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)
            else:
                other_params.append(param)
        
        return [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": other_params, "lr": default_lr},
        ]
    
    def save_pretrained(self, save_path):
        """Save model and processor to directory."""
        self.model.save_pretrained(save_path)
        self.image_processor.save_pretrained(save_path)
    
    @classmethod
    def from_pretrained(cls, load_path, num_classes=None):
        """Load model and processor from directory."""
        model = DetrForObjectDetection.from_pretrained(load_path)
        processor = DetrImageProcessor.from_pretrained(load_path)
        
        # Create wrapper instance
        wrapper = cls.__new__(cls)
        nn.Module.__init__(wrapper)
        wrapper.model = model
        wrapper.image_processor = processor
        wrapper.num_classes = num_classes or model.config.num_labels
        wrapper.id2label = model.config.id2label
        wrapper.label2id = model.config.label2id
        wrapper.model_name = load_path
        wrapper.image_size = processor.size.get("shortest_edge", 256)
        
        return wrapper
