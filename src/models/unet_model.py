"""
Simple UNet Model for semantic segmentation.
Compatible with DeepSpeed distributed training.

A lightweight UNet implementation with skip connections for fast baseline training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Double convolution block: (Conv2d -> BN -> ReLU) x 2
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downsampling block: MaxPool -> DoubleConv
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upsampling block with skip connection.
    Uses bilinear upsampling by default for simplicity and speed.
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        """
        Args:
            x1: Feature map from decoder (to be upsampled)
            x2: Feature map from encoder (skip connection)
        """
        x1 = self.up(x1)
        
        # Handle size mismatch due to odd dimensions
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Output convolution: 1x1 conv to map to num_classes
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Simple UNet implementation for semantic segmentation.
    
    Architecture:
        Encoder: 4 downsampling blocks with skip connections
        Bottleneck: Double convolution
        Decoder: 4 upsampling blocks with skip connections
        
    Args:
        n_channels: Number of input channels (3 for RGB)
        n_classes: Number of output classes (segmentation classes)
        base_filters: Number of filters in first layer (default: 64)
        bilinear: Use bilinear upsampling (default: True for speed)
    """
    def __init__(self, n_channels=3, n_classes=1, base_filters=64, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Encoder
        self.inc = DoubleConv(n_channels, base_filters)
        self.down1 = Down(base_filters, base_filters * 2)
        self.down2 = Down(base_filters * 2, base_filters * 4)
        self.down3 = Down(base_filters * 4, base_filters * 8)
        
        factor = 2 if bilinear else 1
        self.down4 = Down(base_filters * 8, base_filters * 16 // factor)
        
        # Decoder
        self.up1 = Up(base_filters * 16, base_filters * 8 // factor, bilinear)
        self.up2 = Up(base_filters * 8, base_filters * 4 // factor, bilinear)
        self.up3 = Up(base_filters * 4, base_filters * 2 // factor, bilinear)
        self.up4 = Up(base_filters * 2, base_filters, bilinear)
        
        # Output
        self.outc = OutConv(base_filters, n_classes)
    
    def forward(self, x):
        # Encoder with skip connections
        x1 = self.inc(x)      # [B, 64, H, W]
        x2 = self.down1(x1)   # [B, 128, H/2, W/2]
        x3 = self.down2(x2)   # [B, 256, H/4, W/4]
        x4 = self.down3(x3)   # [B, 512, H/8, W/8]
        x5 = self.down4(x4)   # [B, 512, H/16, W/16] or [B, 1024, H/16, W/16]
        
        # Decoder with skip connections
        x = self.up1(x5, x4)  # [B, 256, H/8, W/8]
        x = self.up2(x, x3)   # [B, 128, H/4, W/4]
        x = self.up3(x, x2)   # [B, 64, H/2, W/2]
        x = self.up4(x, x1)   # [B, 64, H, W]
        
        # Output
        logits = self.outc(x)  # [B, n_classes, H, W]
        return logits


class UNetSmall(nn.Module):
    """
    Smaller/faster UNet variant for quick experiments.
    Uses fewer filters (32 base instead of 64).
    """
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        base = 32  # Reduced from 64
        
        # Encoder
        self.inc = DoubleConv(n_channels, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.down3 = Down(base * 4, base * 8)
        
        factor = 2 if bilinear else 1
        self.down4 = Down(base * 8, base * 16 // factor)
        
        # Decoder
        self.up1 = Up(base * 16, base * 8 // factor, bilinear)
        self.up2 = Up(base * 8, base * 4 // factor, bilinear)
        self.up3 = Up(base * 4, base * 2 // factor, bilinear)
        self.up4 = Up(base * 2, base, bilinear)
        
        # Output
        self.outc = OutConv(base, n_classes)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        return logits


class UNetTiny(nn.Module):
    """
    Tiny UNet for fast training on large images.
    - Only 3 encoder/decoder levels (vs 4)
    - 16 base filters (vs 64)
    - ~50x fewer parameters than standard UNet
    - Much lower memory usage for skip connections
    """
    def __init__(self, n_channels=3, n_classes=3, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        base = 16  # Very small base
        
        # Encoder (only 3 levels)
        self.inc = DoubleConv(n_channels, base)           # 16
        self.down1 = Down(base, base * 2)                 # 32
        self.down2 = Down(base * 2, base * 4)             # 64
        
        factor = 2 if bilinear else 1
        self.down3 = Down(base * 4, base * 8 // factor)   # 64 or 128
        
        # Decoder (only 3 levels)
        self.up1 = Up(base * 8, base * 4 // factor, bilinear)  # 32
        self.up2 = Up(base * 4, base * 2 // factor, bilinear)  # 16
        self.up3 = Up(base * 2, base, bilinear)                # 16
        
        # Output
        self.outc = OutConv(base, n_classes)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)       # [B, 16, H, W]
        x2 = self.down1(x1)    # [B, 32, H/2, W/2]
        x3 = self.down2(x2)    # [B, 64, H/4, W/4]
        x4 = self.down3(x3)    # [B, 64, H/8, W/8]
        
        # Decoder
        x = self.up1(x4, x3)   # [B, 32, H/4, W/4]
        x = self.up2(x, x2)    # [B, 16, H/2, W/2]
        x = self.up3(x, x1)    # [B, 16, H, W]
        
        logits = self.outc(x)  # [B, n_classes, H, W]
        return logits


class UNetSegmentor(nn.Module):
    """
    Wrapper around UNet that provides a consistent interface for training.
    Handles loss computation and provides methods compatible with the 
    existing DeepSpeed training loop structure.
    
    Similar to DETRDetector wrapper but for segmentation.
    """
    def __init__(self, n_channels=3, n_classes=3, base_filters=64, 
                 bilinear=True, use_small=False, use_tiny=False):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.base_filters = base_filters
        self.bilinear = bilinear
        self.use_small = use_small
        self.use_tiny = use_tiny
        
        # Choose model variant
        if use_tiny:
            self.model = UNetTiny(n_channels, n_classes, bilinear)
            print(f"[UNetSegmentor] Using UNetTiny (16 base, 3 levels) - FAST")
        elif use_small:
            self.model = UNetSmall(n_channels, n_classes, bilinear)
            print(f"[UNetSegmentor] Using UNetSmall (32 base filters)")
        else:
            self.model = UNet(n_channels, n_classes, base_filters, bilinear)
            print(f"[UNetSegmentor] Using UNet ({base_filters} base filters)")
        
        # Loss function for 3-class segmentation (CE + Dice)
        self.loss_fn = MultiClassSegmentationLoss(n_classes)
        
        print(f"[UNetSegmentor] Channels: {n_channels}, Classes: {n_classes}")
    
    def forward(self, images, masks=None):
        """
        Forward pass through UNet.
        
        Args:
            images: Input images [B, C, H, W]
            masks: Optional ground truth masks [B, H, W] or [B, 1, H, W]
                   When provided, loss is computed and returned
        
        Returns:
            dict containing:
                - 'logits': Raw predictions [B, n_classes, H, W]
                - 'loss': Computed loss (if masks provided)
                - 'pred_masks': Predicted masks [B, H, W] (argmax for multi-class)
        """
        logits = self.model(images)
        
        outputs = {'logits': logits}
        
        # Compute predicted masks (argmax for multi-class)
        outputs['pred_masks'] = torch.argmax(logits, dim=1)
        
        # Compute loss if targets provided
        if masks is not None:
            loss = self.loss_fn(logits, masks)
            outputs['loss'] = loss
        
        return outputs
    
    def save_pretrained(self, save_path):
        """Save model weights in a simple format."""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Save model state dict
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'n_channels': self.n_channels,
            'n_classes': self.n_classes,
            'base_filters': self.base_filters,
            'bilinear': self.bilinear,
            'use_small': self.use_small,
            'use_tiny': self.use_tiny,
        }, os.path.join(save_path, 'unet_model.pt'))
        
        # Save config as JSON for reference
        import json
        config = {
            'n_channels': self.n_channels,
            'n_classes': self.n_classes,
            'base_filters': self.base_filters,
            'bilinear': self.bilinear,
            'use_small': self.use_small,
            'use_tiny': self.use_tiny,
            'model_type': 'UNetTiny' if self.use_tiny else ('UNetSmall' if self.use_small else 'UNet')
        }
        with open(os.path.join(save_path, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def load_pretrained(cls, load_path, device='cpu', use_tiny_fallback=False, use_small_fallback=False):
        """
        Load model from saved weights.
        
        Args:
            load_path: Path to saved model directory
            device: Device to load model on
            use_tiny_fallback: Use UNetTiny if variant info not in checkpoint (for old checkpoints)
            use_small_fallback: Use UNetSmall if variant info not in checkpoint (for old checkpoints)
        """
        import os
        
        checkpoint = torch.load(os.path.join(load_path, 'unet_model.pt'), map_location=device)
        
        # Get model variant info (use fallback for old checkpoints without this info)
        use_tiny = checkpoint.get('use_tiny', use_tiny_fallback)
        use_small = checkpoint.get('use_small', use_small_fallback)
        base_filters = checkpoint.get('base_filters', 64)
        bilinear = checkpoint.get('bilinear', True)
        
        model = cls(
            n_channels=checkpoint['n_channels'],
            n_classes=checkpoint['n_classes'],
            base_filters=base_filters,
            bilinear=bilinear,
            use_small=use_small,
            use_tiny=use_tiny
        )
        model.model.load_state_dict(checkpoint['model_state_dict'])
        
        return model


class MultiClassSegmentationLoss(nn.Module):
    """
    Combined Cross-Entropy + Dice loss for multi-class segmentation.
    """
    def __init__(self, n_classes, ce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        super().__init__()
        self.n_classes = n_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [B, C, H, W] raw predictions (C = n_classes)
            targets: [B, H, W] ground truth class indices
        """
        # Cross entropy loss
        ce_loss = self.ce(logits, targets.long())
        
        # Dice loss (computed per class and averaged)
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets.long(), self.n_classes).permute(0, 3, 1, 2).float()
        
        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()
        
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


# Factory function for convenience
def create_unet_model(n_channels=3, n_classes=1, variant='standard', **kwargs):
    """
    Factory function to create UNet models.
    
    Args:
        n_channels: Input channels (default: 3)
        n_classes: Output classes (default: 1 for binary segmentation)
        variant: 'standard', 'small', or 'wrapped'
        **kwargs: Additional arguments passed to model
    
    Returns:
        UNet model instance
    """
    if variant == 'standard':
        return UNet(n_channels, n_classes, **kwargs)
    elif variant == 'small':
        return UNetSmall(n_channels, n_classes, **kwargs)
    elif variant == 'wrapped':
        return UNetSegmentor(n_channels, n_classes, **kwargs)
    else:
        raise ValueError(f"Unknown variant: {variant}. Use 'standard', 'small', or 'wrapped'.")


if __name__ == "__main__":
    # Quick test
    print("Testing UNet models...")
    
    # Test standard UNet
    # Test standard UNet with 3 classes
    model = UNet(n_channels=3, n_classes=3)
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    print(f"UNet: Input {x.shape} -> Output {out.shape}")
    
    # Test small UNet
    model_small = UNetSmall(n_channels=3, n_classes=3)
    out_small = model_small(x)
    print(f"UNetSmall: Input {x.shape} -> Output {out_small.shape}")
    
    # Test wrapper with 3 classes
    model_wrapped = UNetSegmentor(n_channels=3, n_classes=3)
    masks = torch.randint(0, 3, (2, 256, 256)).long()
    outputs = model_wrapped(x, masks)
    print(f"UNetSegmentor: Loss = {outputs['loss'].item():.4f}")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    total_params_small = sum(p.numel() for p in model_small.parameters())
    print(f"\nParameter counts:")
    print(f"  UNet (64 base): {total_params:,}")
    print(f"  UNetSmall (32 base): {total_params_small:,}")
