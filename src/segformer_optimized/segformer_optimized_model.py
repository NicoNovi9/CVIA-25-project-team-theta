"""
SegFormer Optimized Model for Semantic Segmentation with YOLO-guided ROI.

This module extends the standard SegFormer to perform segmentation only on
bounding box regions detected by YOLO. The approach:
1. Use YOLO to detect spacecraft bounding boxes
2. Extract the bounding box region from the image
3. Resize/pad the region to 512x512:
   - If smaller: pad with black pixels (preserves full resolution)
   - If larger: downsample
4. Run SegFormer segmentation on the prepared region
5. Map predictions back to original image coordinates
6. All pixels outside the bounding box are classified as background

This allows the model to:
- Focus computation on the relevant spacecraft region
- See most bounding boxes at full resolution (most are < 512x512)
- Maintain consistent input size for the segmentation model

Reference: "SegFormer: Simple and Efficient Design for Semantic Segmentation 
           with Transformers" - Xie et al., NeurIPS 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from typing import Optional, Dict, Any, Tuple
import numpy as np


# Model ID mapping for HuggingFace models
SEGFORMER_MODELS = {
    "b0": "nvidia/segformer-b0-finetuned-ade-512-512",
    "b1": "nvidia/segformer-b1-finetuned-ade-512-512",
    "b2": "nvidia/segformer-b2-finetuned-ade-512-512",
    "b3": "nvidia/segformer-b3-finetuned-ade-512-512",
    "b4": "nvidia/segformer-b4-finetuned-ade-512-512",
    "b5": "nvidia/segformer-b5-finetuned-ade-512-512",
}

# Cityscapes pretrained models
SEGFORMER_MODELS_CITYSCAPES = {
    "b0": "nvidia/segformer-b0-finetuned-cityscapes-512-1024",
    "b1": "nvidia/segformer-b1-finetuned-cityscapes-512-1024",
    "b2": "nvidia/segformer-b2-finetuned-cityscapes-512-1024",
    "b3": "nvidia/segformer-b3-finetuned-cityscapes-512-1024",
    "b4": "nvidia/segformer-b4-finetuned-cityscapes-512-1024",
    "b5": "nvidia/segformer-b5-finetuned-cityscapes-512-1024",
}

# ImageNet pretrained (encoder only)
SEGFORMER_MODELS_IMAGENET = {
    "b0": "nvidia/mit-b0",
    "b1": "nvidia/mit-b1",
    "b2": "nvidia/mit-b2",
    "b3": "nvidia/mit-b3",
    "b4": "nvidia/mit-b4",
    "b5": "nvidia/mit-b5",
}


class MultiClassSegmentationLoss(nn.Module):
    """
    Combined Cross-Entropy + Dice loss for multi-class segmentation.
    """
    def __init__(self, n_classes, ce_weight=0.5, dice_weight=0.5, 
                 class_weights=None, label_smoothing=0.0, smooth=1e-6):
        super().__init__()
        self.n_classes = n_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.register_buffer('class_weights', class_weights)
        
        self.ce = nn.CrossEntropyLoss(
            weight=class_weights, 
            label_smoothing=label_smoothing
        )
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [B, C, H, W] raw predictions
            targets: [B, H, W] ground truth class indices
        """
        targets = targets.long()
        
        if logits.shape[2:] != targets.shape[1:]:
            logits = F.interpolate(
                logits, 
                size=targets.shape[1:], 
                mode='bilinear', 
                align_corners=False
            )
        
        ce_loss = self.ce(logits, targets)
        
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, self.n_classes).permute(0, 3, 1, 2).float()
        
        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()
        
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    def __init__(self, n_classes, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.n_classes = n_classes
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, logits, targets):
        if logits.shape[2:] != targets.shape[1:]:
            logits = F.interpolate(logits, size=targets.shape[1:], mode='bilinear', align_corners=False)
        
        ce_loss = F.cross_entropy(logits, targets.long(), reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class BBoxProcessor:
    """
    Handles bounding box extraction, padding/resizing, and inverse mapping.
    
    The key idea:
    - Most spacecraft bounding boxes are smaller than 512x512
    - For these, we pad with black pixels to maintain full resolution
    - Only larger bounding boxes are downsampled
    - After segmentation, we map predictions back to original coordinates
    """
    
    def __init__(self, target_size: int = 512, bbox_expansion: float = 1.1):
        """
        Args:
            target_size: Target size for SegFormer input (default 512)
            bbox_expansion: Factor to expand bounding box (1.1 = 10% expansion)
        """
        self.target_size = target_size
        self.bbox_expansion = bbox_expansion
    
    def expand_bbox(self, bbox: Tuple[int, int, int, int], 
                    img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """
        Expand bounding box by the expansion factor while staying within image bounds.
        
        Args:
            bbox: (x_min, y_min, x_max, y_max)
            img_width: Original image width
            img_height: Original image height
        
        Returns:
            Expanded bbox (x_min, y_min, x_max, y_max)
        """
        x_min, y_min, x_max, y_max = bbox
        
        width = x_max - x_min
        height = y_max - y_min
        
        # Calculate expansion
        expand_w = int(width * (self.bbox_expansion - 1) / 2)
        expand_h = int(height * (self.bbox_expansion - 1) / 2)
        
        # Apply expansion with bounds checking
        x_min = max(0, x_min - expand_w)
        y_min = max(0, y_min - expand_h)
        x_max = min(img_width, x_max + expand_w)
        y_max = min(img_height, y_max + expand_h)
        
        return (x_min, y_min, x_max, y_max)
    
    def extract_and_prepare(self, image: torch.Tensor, bbox: Tuple[int, int, int, int],
                           mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Extract bounding box region and prepare for SegFormer.
        
        - If bbox region < target_size: pad with zeros (black)
        - If bbox region > target_size: downsample
        
        Args:
            image: Input image tensor [C, H, W]
            bbox: Bounding box (x_min, y_min, x_max, y_max)
            mask: Optional mask tensor [H, W]
        
        Returns:
            Dictionary with:
                - 'crop_image': Prepared image [C, target_size, target_size]
                - 'crop_mask': Prepared mask [target_size, target_size] (if mask provided)
                - 'metadata': Info needed for inverse mapping
        """
        x_min, y_min, x_max, y_max = bbox
        
        # Clamp coordinates to image bounds
        _, H, W = image.shape
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(W, x_max)
        y_max = min(H, y_max)
        
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        
        # Extract crop
        crop_image = image[:, y_min:y_max, x_min:x_max]
        crop_mask = mask[y_min:y_max, x_min:x_max] if mask is not None else None
        
        # Determine if we need padding or downsampling
        needs_padding = bbox_width <= self.target_size and bbox_height <= self.target_size
        
        if needs_padding:
            # Pad to target size
            pad_right = self.target_size - bbox_width
            pad_bottom = self.target_size - bbox_height
            
            # Pad image (with zeros = black)
            prepared_image = F.pad(crop_image, (0, pad_right, 0, pad_bottom), value=0)
            
            # Pad mask (with background class = 0)
            if crop_mask is not None:
                prepared_mask = F.pad(crop_mask.unsqueeze(0), (0, pad_right, 0, pad_bottom), value=0).squeeze(0)
            else:
                prepared_mask = None
            
            scale_factor = 1.0
            
        else:
            # Downsample to target size
            scale_x = self.target_size / bbox_width
            scale_y = self.target_size / bbox_height
            scale_factor = min(scale_x, scale_y)
            
            prepared_image = F.interpolate(
                crop_image.unsqueeze(0),
                size=(self.target_size, self.target_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            
            if crop_mask is not None:
                prepared_mask = F.interpolate(
                    crop_mask.unsqueeze(0).unsqueeze(0).float(),
                    size=(self.target_size, self.target_size),
                    mode='nearest'
                ).squeeze(0).squeeze(0).long()
            else:
                prepared_mask = None
        
        metadata = {
            'original_bbox': (x_min, y_min, x_max, y_max),
            'bbox_width': bbox_width,
            'bbox_height': bbox_height,
            'needs_padding': needs_padding,
            'scale_factor': scale_factor,
            'original_image_size': (H, W)
        }
        
        result = {
            'crop_image': prepared_image,
            'metadata': metadata
        }
        
        if prepared_mask is not None:
            result['crop_mask'] = prepared_mask
        
        return result
    
    def inverse_map_prediction(self, prediction: torch.Tensor, 
                               metadata: Dict[str, Any],
                               output_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Map prediction back to original image coordinates.
        
        The prediction is for the bounding box region. We need to:
        1. Remove padding or upsample back to bbox size
        2. Place the prediction in a full-size mask (rest is background)
        
        Args:
            prediction: Predicted mask [H, W] or logits [C, H, W]
            metadata: Metadata from extract_and_prepare
            output_size: Optional output size (H, W), defaults to original image size
        
        Returns:
            Full-size prediction with background outside bbox
        """
        x_min, y_min, x_max, y_max = metadata['original_bbox']
        bbox_width = metadata['bbox_width']
        bbox_height = metadata['bbox_height']
        needs_padding = metadata['needs_padding']
        orig_H, orig_W = metadata['original_image_size']
        
        if output_size is None:
            output_size = (orig_H, orig_W)
        
        is_logits = prediction.dim() == 3  # [C, H, W]
        
        if needs_padding:
            # Remove padding
            if is_logits:
                bbox_pred = prediction[:, :bbox_height, :bbox_width]
            else:
                bbox_pred = prediction[:bbox_height, :bbox_width]
        else:
            # Upsample back to bbox size
            if is_logits:
                bbox_pred = F.interpolate(
                    prediction.unsqueeze(0),
                    size=(bbox_height, bbox_width),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            else:
                bbox_pred = F.interpolate(
                    prediction.unsqueeze(0).unsqueeze(0).float(),
                    size=(bbox_height, bbox_width),
                    mode='nearest'
                ).squeeze(0).squeeze(0).long()
        
        # Create full-size output (initialized with background)
        if is_logits:
            C = prediction.shape[0]
            full_pred = torch.zeros(C, output_size[0], output_size[1], 
                                   device=prediction.device, dtype=prediction.dtype)
            # Set background channel to high value (will be argmax winner for background)
            full_pred[0, :, :] = 10.0  # Large positive value for background
        else:
            full_pred = torch.zeros(output_size[0], output_size[1], 
                                   device=prediction.device, dtype=prediction.dtype)
        
        # Place bbox prediction in the correct location
        # Handle coordinate scaling if output_size != original_image_size
        if output_size != (orig_H, orig_W):
            scale_h = output_size[0] / orig_H
            scale_w = output_size[1] / orig_W
            x_min_scaled = int(x_min * scale_w)
            y_min_scaled = int(y_min * scale_h)
            x_max_scaled = int(x_max * scale_w)
            y_max_scaled = int(y_max * scale_h)
            
            # Resize bbox_pred to scaled size
            scaled_height = y_max_scaled - y_min_scaled
            scaled_width = x_max_scaled - x_min_scaled
            
            if is_logits:
                bbox_pred = F.interpolate(
                    bbox_pred.unsqueeze(0),
                    size=(scaled_height, scaled_width),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
                full_pred[:, y_min_scaled:y_max_scaled, x_min_scaled:x_max_scaled] = bbox_pred
            else:
                bbox_pred = F.interpolate(
                    bbox_pred.unsqueeze(0).unsqueeze(0).float(),
                    size=(scaled_height, scaled_width),
                    mode='nearest'
                ).squeeze(0).squeeze(0).long()
                full_pred[y_min_scaled:y_max_scaled, x_min_scaled:x_max_scaled] = bbox_pred
        else:
            if is_logits:
                full_pred[:, y_min:y_max, x_min:x_max] = bbox_pred
            else:
                full_pred[y_min:y_max, x_min:x_max] = bbox_pred
        
        return full_pred


class SegFormerOptimizedSegmentor(nn.Module):
    """
    SegFormer with YOLO-guided ROI extraction for optimized segmentation.
    
    This model:
    1. Receives pre-extracted bounding box crops (from dataset)
    2. Runs SegFormer on the 512x512 prepared crops
    3. The mapping back to full image is handled externally (in eval/inference)
    
    For training, the dataset provides pre-processed bbox crops.
    For inference, the full pipeline (YOLO + crop + segment + map back) is used.
    """
    
    def __init__(
        self, 
        n_classes: int = 3,
        variant: str = "b2",
        pretrained: Optional[str] = "ade20k",
        freeze_backbone: bool = False,
        dropout: float = 0.1,
        loss_config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        
        self.n_classes = n_classes
        self.variant = variant.lower()
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.dropout_rate = dropout
        self.loss_config = loss_config or {}
        
        if self.variant not in SEGFORMER_MODELS:
            raise ValueError(f"Unknown variant: {self.variant}. Choose from: {list(SEGFORMER_MODELS.keys())}")
        
        try:
            from transformers import SegformerForSemanticSegmentation, SegformerConfig
        except ImportError:
            raise ImportError(
                "transformers library is required for SegFormer. "
                "Install with: pip install transformers"
            )
        
        self._load_model(SegformerForSemanticSegmentation, SegformerConfig)
        self._setup_loss()
        
        print(f"[SegFormerOptimized] Loaded variant: {self.variant}")
        print(f"[SegFormerOptimized] Pretrained: {self.pretrained}")
        print(f"[SegFormerOptimized] Num classes: {self.n_classes}")
        print(f"[SegFormerOptimized] Freeze backbone: {self.freeze_backbone}")
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[SegFormerOptimized] Total params: {total_params:,}")
        print(f"[SegFormerOptimized] Trainable params: {trainable_params:,}")
    
    def _load_model(self, ModelClass, ConfigClass):
        """Load SegFormer model with appropriate pretrained weights."""
        from transformers import SegformerForSemanticSegmentation
        
        if self.pretrained == "ade20k":
            model_id = SEGFORMER_MODELS[self.variant]
            print(f"[SegFormerOptimized] Loading from: {model_id}")
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                model_id,
                num_labels=self.n_classes,
                ignore_mismatched_sizes=True
            )
        elif self.pretrained == "cityscapes":
            model_id = SEGFORMER_MODELS_CITYSCAPES[self.variant]
            print(f"[SegFormerOptimized] Loading from: {model_id}")
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                model_id,
                num_labels=self.n_classes,
                ignore_mismatched_sizes=True
            )
        elif self.pretrained == "imagenet":
            encoder_id = SEGFORMER_MODELS_IMAGENET[self.variant]
            print(f"[SegFormerOptimized] Loading encoder from: {encoder_id}")
            from transformers import SegformerModel
            encoder = SegformerModel.from_pretrained(encoder_id)
            
            config = encoder.config
            config.num_labels = self.n_classes
            self.model = SegformerForSemanticSegmentation(config)
            self.model.segformer.load_state_dict(encoder.state_dict())
            print("[SegFormerOptimized] Encoder weights loaded, decoder initialized randomly")
        else:
            print(f"[SegFormerOptimized] Initializing {self.variant} from scratch")
            from transformers import SegformerConfig
            
            variant_configs = {
                "b0": {"hidden_sizes": [32, 64, 160, 256], "depths": [2, 2, 2, 2], "decoder_hidden_size": 256},
                "b1": {"hidden_sizes": [64, 128, 320, 512], "depths": [2, 2, 2, 2], "decoder_hidden_size": 256},
                "b2": {"hidden_sizes": [64, 128, 320, 512], "depths": [3, 4, 6, 3], "decoder_hidden_size": 768},
                "b3": {"hidden_sizes": [64, 128, 320, 512], "depths": [3, 4, 18, 3], "decoder_hidden_size": 768},
                "b4": {"hidden_sizes": [64, 128, 320, 512], "depths": [3, 8, 27, 3], "decoder_hidden_size": 768},
                "b5": {"hidden_sizes": [64, 128, 320, 512], "depths": [3, 6, 40, 3], "decoder_hidden_size": 768},
            }
            
            config = SegformerConfig(
                num_labels=self.n_classes,
                **variant_configs[self.variant]
            )
            self.model = SegformerForSemanticSegmentation(config)
        
        if self.freeze_backbone:
            for param in self.model.segformer.parameters():
                param.requires_grad = False
            print("[SegFormerOptimized] Backbone frozen")
    
    def _setup_loss(self):
        """Setup loss function based on configuration."""
        loss_type = self.loss_config.get('type', 'ce_dice')
        
        if loss_type == 'ce':
            self.loss_fn = nn.CrossEntropyLoss(
                weight=self._get_class_weights(),
                label_smoothing=self.loss_config.get('label_smoothing', 0.0)
            )
        elif loss_type == 'dice':
            self.loss_fn = MultiClassSegmentationLoss(
                self.n_classes, ce_weight=0.0, dice_weight=1.0
            )
        elif loss_type == 'ce_dice':
            self.loss_fn = MultiClassSegmentationLoss(
                self.n_classes,
                ce_weight=self.loss_config.get('ce_weight', 0.5),
                dice_weight=self.loss_config.get('dice_weight', 0.5),
                class_weights=self.loss_config.get('class_weights'),
                label_smoothing=self.loss_config.get('label_smoothing', 0.0)
            )
        elif loss_type == 'focal':
            self.loss_fn = FocalLoss(
                self.n_classes,
                gamma=self.loss_config.get('focal_gamma', 2.0),
                alpha=self._get_class_weights()
            )
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        print(f"[SegFormerOptimized] Loss function: {loss_type}")
    
    def _get_class_weights(self):
        """Get class weights tensor if specified."""
        weights = self.loss_config.get('class_weights')
        if weights is not None:
            return torch.tensor(weights, dtype=torch.float32)
        return None
    
    def forward(self, images, masks=None):
        """
        Forward pass through SegFormer.
        
        Input images are expected to be pre-processed bbox crops at 512x512.
        
        Args:
            images: Input images [B, C, H, W] (bbox crops, 512x512)
            masks: Optional ground truth masks [B, H, W] for loss computation
        
        Returns:
            dict containing:
                - 'logits': Raw predictions [B, n_classes, H, W]
                - 'pred_masks': Predicted class indices [B, H, W]
                - 'loss': Computed loss (if masks provided)
        """
        outputs = self.model(pixel_values=images)
        logits = outputs.logits
        
        logits_upsampled = F.interpolate(
            logits,
            size=images.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
        result = {
            'logits': logits_upsampled,
            'pred_masks': torch.argmax(logits_upsampled, dim=1)
        }
        
        if masks is not None:
            loss = self.loss_fn(logits_upsampled, masks)
            result['loss'] = loss
        
        return result
    
    def save_pretrained(self, save_path):
        """Save model weights and configuration."""
        os.makedirs(save_path, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'n_classes': self.n_classes,
            'variant': self.variant,
            'pretrained': self.pretrained,
            'freeze_backbone': self.freeze_backbone,
            'dropout_rate': self.dropout_rate,
            'loss_config': self.loss_config,
        }, os.path.join(save_path, 'segformer_optimized_model.pt'))
        
        config = {
            'n_classes': self.n_classes,
            'variant': self.variant,
            'pretrained': self.pretrained,
            'freeze_backbone': self.freeze_backbone,
            'dropout_rate': self.dropout_rate,
            'loss_config': self.loss_config,
            'model_type': 'SegFormerOptimized'
        }
        with open(os.path.join(save_path, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"[SegFormerOptimized] Model saved to {save_path}")
    
    @classmethod
    def load_pretrained(cls, load_path, device='cpu'):
        """Load model from saved weights."""
        checkpoint = torch.load(
            os.path.join(load_path, 'segformer_optimized_model.pt'),
            map_location=device
        )
        
        model = cls(
            n_classes=checkpoint['n_classes'],
            variant=checkpoint['variant'],
            pretrained=None,
            freeze_backbone=checkpoint.get('freeze_backbone', False),
            dropout=checkpoint.get('dropout_rate', 0.1),
            loss_config=checkpoint.get('loss_config', {})
        )
        
        model.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[SegFormerOptimized] Model loaded from {load_path}")
        
        return model


def list_available_variants():
    """List available SegFormer variants with descriptions."""
    variants = {
        "b0": "3.7M params - Fastest, good for prototyping",
        "b1": "13.7M params - Light model",
        "b2": "24.7M params - Recommended balance of speed/accuracy",
        "b3": "44.6M params - Higher accuracy",
        "b4": "61.4M params - High accuracy",
        "b5": "81.4M params - Highest accuracy, slowest",
    }
    return variants


if __name__ == "__main__":
    print("Testing SegFormer Optimized models...")
    print("\nAvailable variants:")
    for var, desc in list_available_variants().items():
        print(f"  {var}: {desc}")
    
    print("\nLoading SegFormerOptimized-b0...")
    model = SegFormerOptimizedSegmentor(
        n_classes=3,
        variant="b0",
        pretrained="ade20k"
    )
    
    # Test forward pass with bbox crop
    x = torch.randn(2, 3, 512, 512)  # Simulated bbox crops
    masks = torch.randint(0, 3, (2, 512, 512)).long()
    
    print("\nTesting forward pass...")
    outputs = model(x, masks)
    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Pred masks shape: {outputs['pred_masks'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")
    
    # Test BBoxProcessor
    print("\nTesting BBoxProcessor...")
    processor = BBoxProcessor(target_size=512, bbox_expansion=1.1)
    
    # Test with small bbox (should pad)
    small_image = torch.randn(3, 1024, 1024)
    small_mask = torch.randint(0, 3, (1024, 1024)).long()
    small_bbox = (100, 100, 400, 350)  # 300x250, smaller than 512
    
    result = processor.extract_and_prepare(small_image, small_bbox, small_mask)
    print(f"Small bbox ({small_bbox[2]-small_bbox[0]}x{small_bbox[3]-small_bbox[1]}):")
    print(f"  Needs padding: {result['metadata']['needs_padding']}")
    print(f"  Crop image shape: {result['crop_image'].shape}")
    
    # Test with large bbox (should downsample)
    large_bbox = (100, 100, 750, 700)  # 650x600, larger than 512
    
    result = processor.extract_and_prepare(small_image, large_bbox, small_mask)
    print(f"Large bbox ({large_bbox[2]-large_bbox[0]}x{large_bbox[3]-large_bbox[1]}):")
    print(f"  Needs padding: {result['metadata']['needs_padding']}")
    print(f"  Crop image shape: {result['crop_image'].shape}")
    
    # Test save/load
    print("\nTesting save/load...")
    model.save_pretrained("/tmp/segformer_optimized_test")
    model_loaded = SegFormerOptimizedSegmentor.load_pretrained("/tmp/segformer_optimized_test")
    outputs_loaded = model_loaded(x)
    print(f"Loaded model output shape: {outputs_loaded['logits'].shape}")
    
    print("\n✅ All tests passed!")
