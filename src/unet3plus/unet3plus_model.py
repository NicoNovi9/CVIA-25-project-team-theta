"""
UNet3+ with Pretrained Backbone for Semantic Segmentation.
Uses ConvNeXt or HRNet backbone from timm for pretrained feature extraction.

UNet3+ introduces full-scale skip connections to capture multi-scale features,
combining deep semantic features with fine-grained details from all encoder levels.

References:
    - UNet3+: https://arxiv.org/abs/2004.08790
    - timm: https://github.com/huggingface/pytorch-image-models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import List, Optional, Tuple


class ConvBNReLU(nn.Module):
    """Convolution + BatchNorm + ReLU block."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                              padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class FullScaleSkipConnection(nn.Module):
    """
    Full-scale skip connection module for UNet3+.
    Aggregates features from all encoder and decoder levels at a target scale.
    """
    def __init__(self, encoder_channels: List[int], decoder_channels: List[int],
                 target_level: int, cat_channels: int = 64, up_sample_mode: str = 'bilinear'):
        """
        Args:
            encoder_channels: List of encoder channels at each level [e1, e2, e3, e4, e5]
            decoder_channels: List of decoder channels at each level [d4, d3, d2, d1]
            target_level: Which decoder level we are building (4=deepest, 1=shallowest)
            cat_channels: Output channels per skip connection (default 64)
            up_sample_mode: Upsampling mode ('bilinear' or 'nearest')
        """
        super().__init__()
        self.target_level = target_level
        self.cat_channels = cat_channels
        self.up_sample_mode = up_sample_mode
        
        self.encoder_convs = nn.ModuleList()
        self.encoder_scales = []  # Relative scale factors for encoders
        
        # Number of encoder levels
        num_encoder_levels = len(encoder_channels)
        
        # For each encoder level, create a conv to project to cat_channels
        for i, enc_ch in enumerate(encoder_channels):
            self.encoder_convs.append(ConvBNReLU(enc_ch, cat_channels, 3, 1, 1))
            # Calculate scale factor: higher levels need upsampling, lower need downsampling
            # Level i has spatial size H/2^i, target has size H/2^target_level
            # Scale = 2^(target_level - i)
            scale = 2 ** (target_level - i)
            self.encoder_scales.append(scale)
        
        # For decoder levels (only levels deeper than target_level)
        self.decoder_convs = nn.ModuleList()
        self.decoder_scales = []
        
        # decoder_channels is [d4, d3, d2, d1] for 5-level encoder
        # d4 is from encoder level 5, d3 from combining, etc.
        for i, dec_ch in enumerate(decoder_channels):
            dec_level = num_encoder_levels - 1 - i  # d4 = level 4, d3 = level 3, etc.
            if dec_level > target_level:
                self.decoder_convs.append(ConvBNReLU(dec_ch, cat_channels, 3, 1, 1))
                scale = 2 ** (target_level - dec_level)  # Will be < 1, meaning upsample
                self.decoder_scales.append(scale)
        
        # Total number of concatenated features
        num_connections = len(encoder_channels) + len(self.decoder_convs)
        total_channels = cat_channels * num_connections
        
        # Final fusion convolution
        self.fusion_conv = ConvBNReLU(total_channels, cat_channels * num_encoder_levels, 3, 1, 1)
    
    def _resize(self, x: torch.Tensor, target_size: Tuple[int, int], scale: float) -> torch.Tensor:
        """Resize feature map to target size."""
        if scale == 1.0:
            return x
        elif scale > 1:
            # Upsample
            return F.interpolate(x, size=target_size, mode=self.up_sample_mode, 
                               align_corners=True if self.up_sample_mode == 'bilinear' else None)
        else:
            # Downsample
            return F.interpolate(x, size=target_size, mode=self.up_sample_mode,
                               align_corners=True if self.up_sample_mode == 'bilinear' else None)
    
    def forward(self, encoder_features: List[torch.Tensor], 
                decoder_features: List[torch.Tensor] = None,
                target_size: Tuple[int, int] = None) -> torch.Tensor:
        """
        Args:
            encoder_features: Features from encoder levels [e1, e2, e3, e4, e5]
            decoder_features: Features from previously computed decoder levels
            target_size: Target spatial size (H, W)
        """
        if target_size is None:
            target_size = encoder_features[self.target_level].shape[2:]
        
        aggregated = []
        
        # Process encoder features
        for i, (feat, conv, scale) in enumerate(zip(encoder_features, self.encoder_convs, self.encoder_scales)):
            x = conv(feat)
            x = self._resize(x, target_size, scale)
            aggregated.append(x)
        
        # Process decoder features (if any)
        if decoder_features is not None:
            for feat, conv, scale in zip(decoder_features, self.decoder_convs, self.decoder_scales):
                x = conv(feat)
                x = self._resize(x, target_size, scale)
                aggregated.append(x)
        
        # Concatenate and fuse
        x = torch.cat(aggregated, dim=1)
        x = self.fusion_conv(x)
        
        return x


class UNet3PlusBackbone(nn.Module):
    """
    UNet3+ with pretrained backbone from timm.
    
    Supports:
        - ConvNeXt (convnext_tiny, convnext_small, convnext_base)
        - HRNet (hrnet_w18, hrnet_w32, hrnet_w48)
        - EfficientNet (efficientnet_b0 through b7)
        - ResNet (resnet18, resnet34, resnet50, resnet101)
    
    Architecture:
        - Encoder: Pretrained backbone with 5 feature scales
        - Decoder: Full-scale skip connections (UNet3+ style)
        - Deep supervision: Optional auxiliary outputs at each decoder level
    """
    
    SUPPORTED_BACKBONES = {
        'convnext_tiny': {'pretrained': 'convnext_tiny.fb_in22k_ft_in1k', 'channels': [96, 192, 384, 768]},
        'convnext_small': {'pretrained': 'convnext_small.fb_in22k_ft_in1k', 'channels': [96, 192, 384, 768]},
        'convnext_base': {'pretrained': 'convnext_base.fb_in22k_ft_in1k', 'channels': [128, 256, 512, 1024]},
        'hrnet_w18': {'pretrained': 'hrnet_w18', 'channels': [18, 36, 72, 144]},
        'hrnet_w32': {'pretrained': 'hrnet_w32', 'channels': [32, 64, 128, 256]},
        'hrnet_w48': {'pretrained': 'hrnet_w48', 'channels': [48, 96, 192, 384]},
        'efficientnet_b0': {'pretrained': 'efficientnet_b0', 'channels': [16, 24, 40, 112, 320]},
        'efficientnet_b3': {'pretrained': 'efficientnet_b3', 'channels': [24, 32, 48, 136, 384]},
        'resnet34': {'pretrained': 'resnet34', 'channels': [64, 64, 128, 256, 512]},
        'resnet50': {'pretrained': 'resnet50', 'channels': [64, 256, 512, 1024, 2048]},
    }
    
    def __init__(self, 
                 backbone_name: str = 'convnext_tiny',
                 n_classes: int = 3,
                 pretrained: bool = True,
                 cat_channels: int = 64,
                 deep_supervision: bool = False,
                 freeze_backbone: bool = False):
        """
        Args:
            backbone_name: Name of the backbone (see SUPPORTED_BACKBONES)
            n_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            cat_channels: Channels per skip connection
            deep_supervision: Enable deep supervision outputs
            freeze_backbone: Freeze backbone weights
        """
        super().__init__()
        
        self.backbone_name = backbone_name
        self.n_classes = n_classes
        self.deep_supervision = deep_supervision
        self.cat_channels = cat_channels
        
        if backbone_name not in self.SUPPORTED_BACKBONES:
            raise ValueError(f"Backbone {backbone_name} not supported. "
                           f"Choose from: {list(self.SUPPORTED_BACKBONES.keys())}")
        
        backbone_info = self.SUPPORTED_BACKBONES[backbone_name]
        
        # Create backbone with feature extraction
        print(f"[UNet3+] Loading backbone: {backbone_name} (pretrained={pretrained})")
        self.backbone = timm.create_model(
            backbone_info['pretrained'] if pretrained else backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3) if len(backbone_info['channels']) == 4 else (0, 1, 2, 3, 4)
        )
        
        # Get actual encoder channels from backbone
        self.encoder_channels = self.backbone.feature_info.channels()
        print(f"[UNet3+] Encoder channels: {self.encoder_channels}")
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("[UNet3+] Backbone frozen")
        
        # Initial stem to normalize input to 64 channels (for consistent first level)
        # Most backbones include a stem, so we use the backbone's first feature
        num_levels = len(self.encoder_channels)
        
        # Calculate decoder channels (5 * cat_channels for each level with 5 skip connections)
        self.decoder_channels = cat_channels * num_levels
        
        # Build decoder with full-scale skip connections
        # We build from deepest to shallowest
        self.decoder_convs = nn.ModuleList()
        
        for level in range(num_levels - 2, -1, -1):  # From second deepest to shallowest
            in_ch = self.encoder_channels[level] + self.decoder_channels * (num_levels - 2 - level + 1) \
                    if level < num_levels - 1 else self.encoder_channels[level]
            
            # For simplified UNet3+, we use standard skip connections with multi-scale aggregation
            self.decoder_convs.append(
                nn.Sequential(
                    ConvBNReLU(in_ch if level == num_levels - 2 else self.decoder_channels + self.encoder_channels[level], 
                              self.decoder_channels, 3, 1, 1),
                    ConvBNReLU(self.decoder_channels, self.decoder_channels, 3, 1, 1)
                )
            )
        
        # Build simpler UNet3+ decoder
        self._build_unet3plus_decoder(num_levels, cat_channels)
        
        # Output head
        self.output_conv = nn.Conv2d(self.decoder_channels, n_classes, kernel_size=1)
        
        # Deep supervision heads
        if deep_supervision:
            self.deep_outputs = nn.ModuleList([
                nn.Conv2d(self.decoder_channels, n_classes, kernel_size=1)
                for _ in range(num_levels - 1)
            ])
        
        print(f"[UNet3+] Model created: {n_classes} classes, decoder_channels={self.decoder_channels}")
    
    def _build_unet3plus_decoder(self, num_levels: int, cat_channels: int):
        """Build UNet3+ style decoder with full-scale skip connections."""
        # For each decoder level, we aggregate from all encoder levels
        # Using simpler approach: at each level, combine upsampled deeper level + same-scale encoder
        
        # Clear the previous convs
        self.decoder_convs = nn.ModuleList()
        
        # Upsampling modules
        self.up_convs = nn.ModuleList()
        
        # From deepest (level num_levels-1) to shallowest (level 0)
        for i in range(num_levels - 1):
            # Input from deeper decoder level (or bottleneck) + encoder at this level
            if i == 0:
                # First decoder level: takes bottleneck output
                in_ch = self.encoder_channels[-1]
            else:
                in_ch = self.decoder_channels
            
            self.up_convs.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    ConvBNReLU(in_ch, self.decoder_channels, 3, 1, 1)
                )
            )
            
            # Combine with encoder features at current level
            enc_level = num_levels - 1 - (i + 1)  # Encoder level index
            combined_ch = self.decoder_channels + self.encoder_channels[enc_level]
            
            self.decoder_convs.append(
                nn.Sequential(
                    ConvBNReLU(combined_ch, self.decoder_channels, 3, 1, 1),
                    ConvBNReLU(self.decoder_channels, self.decoder_channels, 3, 1, 1)
                )
            )
        
        # Multi-scale feature aggregation (UNet3+ key feature)
        # For final output, aggregate features from all decoder levels
        self.multi_scale_agg = nn.ModuleList()
        for i in range(num_levels - 1):
            scale = 2 ** i
            self.multi_scale_agg.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True) if scale > 1 else nn.Identity(),
                    ConvBNReLU(self.decoder_channels, cat_channels, 1, 1, 0)
                )
            )
        
        # Final fusion
        self.final_fusion = ConvBNReLU(cat_channels * (num_levels - 1), self.decoder_channels, 3, 1, 1)
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass.
        
        Args:
            x: Input images [B, 3, H, W]
        
        Returns:
            dict with 'logits' and optionally 'deep_outputs'
        """
        input_size = x.shape[2:]
        
        # Encoder: extract multi-scale features
        encoder_features = self.backbone(x)
        
        # Decoder: progressive upsampling with skip connections
        decoder_features = []
        x = encoder_features[-1]  # Start from bottleneck
        
        for i, (up_conv, dec_conv) in enumerate(zip(self.up_convs, self.decoder_convs)):
            # Upsample
            x = up_conv(x)
            
            # Get encoder feature at corresponding level
            enc_level = len(encoder_features) - 2 - i
            enc_feat = encoder_features[enc_level]
            
            # Handle size mismatch
            if x.shape[2:] != enc_feat.shape[2:]:
                x = F.interpolate(x, size=enc_feat.shape[2:], mode='bilinear', align_corners=True)
            
            # Concatenate and process
            x = torch.cat([x, enc_feat], dim=1)
            x = dec_conv(x)
            decoder_features.append(x)
        
        # Multi-scale aggregation (UNet3+ style)
        target_size = decoder_features[-1].shape[2:]
        multi_scale_feats = []
        for feat, agg in zip(decoder_features, self.multi_scale_agg):
            feat_scaled = agg(feat)
            if feat_scaled.shape[2:] != target_size:
                feat_scaled = F.interpolate(feat_scaled, size=target_size, mode='bilinear', align_corners=True)
            multi_scale_feats.append(feat_scaled)
        
        x = torch.cat(multi_scale_feats, dim=1)
        x = self.final_fusion(x)
        
        # Upsample to input resolution
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        # Output
        logits = self.output_conv(x)
        
        outputs = {'logits': logits}
        
        # Deep supervision
        if self.deep_supervision and self.training:
            deep_logits = []
            for feat, head in zip(decoder_features, self.deep_outputs):
                feat_up = F.interpolate(feat, size=input_size, mode='bilinear', align_corners=True)
                deep_logits.append(head(feat_up))
            outputs['deep_outputs'] = deep_logits
        
        return outputs


class UNet3PlusSegmentor(nn.Module):
    """
    Wrapper around UNet3Plus that provides a consistent interface for training.
    Handles loss computation and provides methods compatible with the 
    existing DeepSpeed training loop structure.
    """
    def __init__(self, 
                 backbone_name: str = 'convnext_tiny',
                 n_channels: int = 3,
                 n_classes: int = 3,
                 pretrained: bool = True,
                 cat_channels: int = 64,
                 deep_supervision: bool = False,
                 freeze_backbone: bool = False):
        super().__init__()
        
        self.backbone_name = backbone_name
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.pretrained = pretrained
        self.cat_channels = cat_channels
        self.deep_supervision = deep_supervision
        self.freeze_backbone = freeze_backbone
        
        # Create model
        self.model = UNet3PlusBackbone(
            backbone_name=backbone_name,
            n_classes=n_classes,
            pretrained=pretrained,
            cat_channels=cat_channels,
            deep_supervision=deep_supervision,
            freeze_backbone=freeze_backbone
        )
        
        # Loss function for 3-class segmentation (CE + Dice)
        self.loss_fn = MultiClassSegmentationLoss(n_classes, deep_supervision=deep_supervision)
        
        print(f"[UNet3PlusSegmentor] Created with {backbone_name} backbone")
        print(f"[UNet3PlusSegmentor] Classes: {n_classes}, Deep supervision: {deep_supervision}")
    
    def forward(self, images: torch.Tensor, masks: torch.Tensor = None) -> dict:
        """
        Forward pass.
        
        Args:
            images: Input images [B, C, H, W]
            masks: Optional ground truth masks [B, H, W]
        
        Returns:
            dict containing:
                - 'logits': Raw predictions [B, n_classes, H, W]
                - 'loss': Computed loss (if masks provided)
                - 'pred_masks': Predicted masks [B, H, W]
        """
        outputs = self.model(images)
        
        # Get predicted masks
        outputs['pred_masks'] = torch.argmax(outputs['logits'], dim=1)
        
        # Compute loss if targets provided
        if masks is not None:
            loss = self.loss_fn(outputs, masks)
            outputs['loss'] = loss
        
        return outputs
    
    def save_pretrained(self, save_path: str):
        """Save model weights."""
        import os
        import json
        
        os.makedirs(save_path, exist_ok=True)
        
        # Save model state dict
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'backbone_name': self.backbone_name,
            'n_channels': self.n_channels,
            'n_classes': self.n_classes,
            'cat_channels': self.cat_channels,
            'deep_supervision': self.deep_supervision,
            'freeze_backbone': self.freeze_backbone,
        }, os.path.join(save_path, 'unet3plus_model.pt'))
        
        # Save config as JSON
        config = {
            'backbone_name': self.backbone_name,
            'n_channels': self.n_channels,
            'n_classes': self.n_classes,
            'cat_channels': self.cat_channels,
            'deep_supervision': self.deep_supervision,
            'freeze_backbone': self.freeze_backbone,
            'model_type': 'UNet3PlusBackbone'
        }
        with open(os.path.join(save_path, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"[UNet3PlusSegmentor] Model saved to {save_path}")
    
    @classmethod
    def load_pretrained(cls, load_path: str, device: str = 'cpu'):
        """Load model from saved weights."""
        import os
        
        checkpoint = torch.load(os.path.join(load_path, 'unet3plus_model.pt'), map_location=device)
        
        # Create model (pretrained=False since we're loading weights)
        model = cls(
            backbone_name=checkpoint['backbone_name'],
            n_channels=checkpoint['n_channels'],
            n_classes=checkpoint['n_classes'],
            pretrained=False,  # Don't load pretrained backbone weights
            cat_channels=checkpoint['cat_channels'],
            deep_supervision=checkpoint.get('deep_supervision', False),
            freeze_backbone=checkpoint.get('freeze_backbone', False)
        )
        
        model.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[UNet3PlusSegmentor] Model loaded from {load_path}")
        
        return model


class MultiClassSegmentationLoss(nn.Module):
    """
    Combined Cross-Entropy + Dice loss for multi-class segmentation.
    Supports deep supervision.
    """
    def __init__(self, n_classes: int, ce_weight: float = 0.5, dice_weight: float = 0.5, 
                 smooth: float = 1e-6, deep_supervision: bool = False):
        super().__init__()
        self.n_classes = n_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.deep_supervision = deep_supervision
        self.ce = nn.CrossEntropyLoss()
    
    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute combined CE + Dice loss."""
        # Cross entropy loss
        ce_loss = self.ce(logits, targets.long())
        
        # Dice loss
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets.long(), self.n_classes).permute(0, 3, 1, 2).float()
        
        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()
        
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss
    
    def forward(self, outputs: dict, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss with optional deep supervision.
        
        Args:
            outputs: dict with 'logits' and optionally 'deep_outputs'
            targets: Ground truth [B, H, W]
        """
        main_loss = self._compute_loss(outputs['logits'], targets)
        
        if self.deep_supervision and 'deep_outputs' in outputs:
            deep_losses = []
            for deep_logits in outputs['deep_outputs']:
                deep_losses.append(self._compute_loss(deep_logits, targets))
            
            # Weight deep supervision losses less
            deep_loss = sum(deep_losses) / len(deep_losses) * 0.4
            return main_loss + deep_loss
        
        return main_loss


def list_available_backbones():
    """List all available backbone options."""
    print("Available backbones for UNet3+:")
    for name, info in UNet3PlusBackbone.SUPPORTED_BACKBONES.items():
        print(f"  - {name}: channels={info['channels']}")


if __name__ == "__main__":
    print("Testing UNet3+ with pretrained backbone...")
    
    # List available backbones
    list_available_backbones()
    print()
    
    # Test with ConvNeXt-Tiny
    print("Testing ConvNeXt-Tiny backbone...")
    model = UNet3PlusSegmentor(
        backbone_name='convnext_tiny',
        n_classes=3,
        pretrained=True,
        deep_supervision=False
    )
    
    # Test forward pass
    x = torch.randn(2, 3, 256, 256)
    masks = torch.randint(0, 3, (2, 256, 256)).long()
    
    outputs = model(x, masks)
    print(f"Input: {x.shape}")
    print(f"Logits: {outputs['logits'].shape}")
    print(f"Pred masks: {outputs['pred_masks'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameter counts:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Test save/load
    print("\nTesting save/load...")
    model.save_pretrained("/tmp/test_unet3plus")
    model_loaded = UNet3PlusSegmentor.load_pretrained("/tmp/test_unet3plus")
    outputs_loaded = model_loaded(x)
    print(f"Loaded model output shape: {outputs_loaded['logits'].shape}")
    print("Save/load test passed!")
