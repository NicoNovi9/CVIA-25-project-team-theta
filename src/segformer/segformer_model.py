"""
SegFormer Model for Semantic Segmentation.
Uses NVIDIA's SegFormer architecture from transformers library.

SegFormer is a hierarchical Transformer encoder with a lightweight MLP decoder.
It provides excellent accuracy/efficiency trade-off for semantic segmentation.

Reference: "SegFormer: Simple and Efficient Design for Semantic Segmentation 
           with Transformers" - Xie et al., NeurIPS 2021

Variants:
    - b0: 3.7M params, fastest
    - b1: 13.7M params
    - b2: 24.7M params (recommended)
    - b3: 44.6M params
    - b4: 61.4M params
    - b5: 81.4M params, highest accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from typing import Optional, Dict, Any


# Model ID mapping for HuggingFace models
SEGFORMER_MODELS = {
    "b0": "nvidia/segformer-b0-finetuned-ade-512-512",
    "b1": "nvidia/segformer-b1-finetuned-ade-512-512",
    "b2": "nvidia/segformer-b2-finetuned-ade-512-512",
    "b3": "nvidia/segformer-b3-finetuned-ade-512-512",
    "b4": "nvidia/segformer-b4-finetuned-ade-512-512",
    "b5": "nvidia/segformer-b5-finetuned-ade-512-512",
}

# Cityscapes pretrained models (higher resolution)
SEGFORMER_MODELS_CITYSCAPES = {
    "b0": "nvidia/segformer-b0-finetuned-cityscapes-512-1024",
    "b1": "nvidia/segformer-b1-finetuned-cityscapes-512-1024",
    "b2": "nvidia/segformer-b2-finetuned-cityscapes-512-1024",
    "b3": "nvidia/segformer-b3-finetuned-cityscapes-512-1024",
    "b4": "nvidia/segformer-b4-finetuned-cityscapes-512-1024",
    "b5": "nvidia/segformer-b5-finetuned-cityscapes-512-1024",
}

# ImageNet pretrained (encoder only, no decoder)
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
    Same as UNet for consistency.
    """
    def __init__(self, n_classes, ce_weight=0.5, dice_weight=0.5, 
                 class_weights=None, label_smoothing=0.0, smooth=1e-6):
        super().__init__()
        self.n_classes = n_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        
        # Class weights for imbalanced data
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
            logits: [B, C, H, W] raw predictions (C = n_classes)
            targets: [B, H, W] ground truth class indices
        """
        # Ensure targets are long type
        targets = targets.long()
        
        # Handle size mismatch (SegFormer may output different resolution)
        if logits.shape[2:] != targets.shape[1:]:
            logits = F.interpolate(
                logits, 
                size=targets.shape[1:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # Cross entropy loss
        ce_loss = self.ce(logits, targets)
        
        # Dice loss (computed per class and averaged)
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


class SegFormerSegmentor(nn.Module):
    """
    SegFormer wrapper for semantic segmentation with consistent training interface.
    
    This wrapper:
    - Loads pretrained SegFormer from HuggingFace transformers
    - Replaces the classification head for custom number of classes
    - Provides loss computation and consistent output format
    - Supports saving/loading checkpoints
    
    Args:
        n_classes: Number of segmentation classes (default: 3)
        variant: Model size 'b0' to 'b5' (default: 'b2')
        pretrained: Pretrained weights source ('ade20k', 'cityscapes', 'imagenet', None)
        freeze_backbone: Freeze encoder weights for fine-tuning
        dropout: Dropout rate for decoder
        loss_config: Dictionary with loss configuration
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
        
        # Validate variant
        if self.variant not in SEGFORMER_MODELS:
            raise ValueError(f"Unknown variant: {self.variant}. Choose from: {list(SEGFORMER_MODELS.keys())}")
        
        # Import transformers here to avoid import errors if not installed
        try:
            from transformers import SegformerForSemanticSegmentation, SegformerConfig
        except ImportError:
            raise ImportError(
                "transformers library is required for SegFormer. "
                "Install with: pip install transformers"
            )
        
        # Load model
        self._load_model(SegformerForSemanticSegmentation, SegformerConfig)
        
        # Setup loss function
        self._setup_loss()
        
        # Print model info
        print(f"[SegFormer] Loaded variant: {self.variant}")
        print(f"[SegFormer] Pretrained: {self.pretrained}")
        print(f"[SegFormer] Num classes: {self.n_classes}")
        print(f"[SegFormer] Freeze backbone: {self.freeze_backbone}")
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[SegFormer] Total params: {total_params:,}")
        print(f"[SegFormer] Trainable params: {trainable_params:,}")
    
    def _load_model(self, ModelClass, ConfigClass):
        """Load SegFormer model with appropriate pretrained weights."""
        from transformers import SegformerForSemanticSegmentation
        
        if self.pretrained == "ade20k":
            model_id = SEGFORMER_MODELS[self.variant]
            print(f"[SegFormer] Loading from: {model_id}")
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                model_id,
                num_labels=self.n_classes,
                ignore_mismatched_sizes=True  # Allow different num_classes
            )
        elif self.pretrained == "cityscapes":
            model_id = SEGFORMER_MODELS_CITYSCAPES[self.variant]
            print(f"[SegFormer] Loading from: {model_id}")
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                model_id,
                num_labels=self.n_classes,
                ignore_mismatched_sizes=True
            )
        elif self.pretrained == "imagenet":
            # Load encoder only, initialize decoder from scratch
            encoder_id = SEGFORMER_MODELS_IMAGENET[self.variant]
            print(f"[SegFormer] Loading encoder from: {encoder_id}")
            from transformers import SegformerModel
            encoder = SegformerModel.from_pretrained(encoder_id)
            
            # Create full model with random decoder
            config = encoder.config
            config.num_labels = self.n_classes
            self.model = SegformerForSemanticSegmentation(config)
            
            # Copy encoder weights
            self.model.segformer.load_state_dict(encoder.state_dict())
            print("[SegFormer] Encoder weights loaded, decoder initialized randomly")
        else:
            # Random initialization
            print(f"[SegFormer] Initializing {self.variant} from scratch")
            from transformers import SegformerConfig
            
            # Get config for variant
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
        
        # Freeze backbone if requested
        if self.freeze_backbone:
            for param in self.model.segformer.parameters():
                param.requires_grad = False
            print("[SegFormer] Backbone frozen")
    
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
        
        print(f"[SegFormer] Loss function: {loss_type}")
    
    def _get_class_weights(self):
        """Get class weights tensor if specified."""
        weights = self.loss_config.get('class_weights')
        if weights is not None:
            return torch.tensor(weights, dtype=torch.float32)
        return None
    
    def forward(self, images, masks=None):
        """
        Forward pass through SegFormer.
        
        Args:
            images: Input images [B, C, H, W]
            masks: Optional ground truth masks [B, H, W] for loss computation
        
        Returns:
            dict containing:
                - 'logits': Raw predictions [B, n_classes, H, W]
                - 'pred_masks': Predicted class indices [B, H, W]
                - 'loss': Computed loss (if masks provided)
        """
        # Forward through SegFormer
        outputs = self.model(pixel_values=images)
        logits = outputs.logits  # [B, n_classes, H/4, W/4]
        
        # Upsample to input resolution
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
        
        # Compute loss if targets provided
        if masks is not None:
            loss = self.loss_fn(logits_upsampled, masks)
            result['loss'] = loss
        
        return result
    
    def save_pretrained(self, save_path):
        """Save model weights and configuration."""
        os.makedirs(save_path, exist_ok=True)
        
        # Save model weights
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'n_classes': self.n_classes,
            'variant': self.variant,
            'pretrained': self.pretrained,
            'freeze_backbone': self.freeze_backbone,
            'dropout_rate': self.dropout_rate,
            'loss_config': self.loss_config,
        }, os.path.join(save_path, 'segformer_model.pt'))
        
        # Save config as JSON for reference
        config = {
            'n_classes': self.n_classes,
            'variant': self.variant,
            'pretrained': self.pretrained,
            'freeze_backbone': self.freeze_backbone,
            'dropout_rate': self.dropout_rate,
            'loss_config': self.loss_config,
            'model_type': 'SegFormer'
        }
        with open(os.path.join(save_path, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"[SegFormer] Model saved to {save_path}")
    
    @classmethod
    def load_pretrained(cls, load_path, device='cpu'):
        """Load model from saved weights."""
        checkpoint = torch.load(
            os.path.join(load_path, 'segformer_model.pt'),
            map_location=device
        )
        
        # Create model with saved configuration
        model = cls(
            n_classes=checkpoint['n_classes'],
            variant=checkpoint['variant'],
            pretrained=None,  # Don't load pretrained again
            freeze_backbone=checkpoint.get('freeze_backbone', False),
            dropout=checkpoint.get('dropout_rate', 0.1),
            loss_config=checkpoint.get('loss_config', {})
        )
        
        # Load weights
        model.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[SegFormer] Model loaded from {load_path}")
        
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
    # Quick test
    print("Testing SegFormer models...")
    print("\nAvailable variants:")
    for var, desc in list_available_variants().items():
        print(f"  {var}: {desc}")
    
    # Test b0 model (smallest)
    print("\nLoading SegFormer-b0...")
    model = SegFormerSegmentor(
        n_classes=3,
        variant="b0",
        pretrained="ade20k"
    )
    
    # Test forward pass
    x = torch.randn(2, 3, 512, 512)
    masks = torch.randint(0, 3, (2, 512, 512)).long()
    
    print("\nTesting forward pass...")
    outputs = model(x, masks)
    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Pred masks shape: {outputs['pred_masks'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")
    
    # Test save/load
    print("\nTesting save/load...")
    model.save_pretrained("/tmp/segformer_test")
    model_loaded = SegFormerSegmentor.load_pretrained("/tmp/segformer_test")
    outputs_loaded = model_loaded(x)
    print(f"Loaded model output shape: {outputs_loaded['logits'].shape}")
    
    print("\n✅ All tests passed!")
