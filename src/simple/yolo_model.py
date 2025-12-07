import torch
import torch.nn as nn
import os
import shutil


class YOLODetector(nn.Module):
    """
    Wrapper for YOLOv8 backbone that's compatible with train.py
    Uses pre-trained YOLO architecture with custom classification and bbox heads
    """
    def __init__(self, num_classes, model_size='n', pretrained=True):
        super().__init__()
        
        try:
            from ultralytics import YOLO

            os.environ["YOLO_HOME"] = "model_weights"

            yolo = self.load_yolo(model_size=model_size, pretrained=pretrained)

            self.backbone = yolo.model.model[:10]
            
            # Dynamically determine feature dimension from backbone output
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 640, 640)
                dummy_output = self.backbone(dummy_input)
                if isinstance(dummy_output, (list, tuple)):
                    dummy_output = dummy_output[-1]
                feat_dim = dummy_output.shape[1]
            
            print(f"YOLO-{model_size} backbone output channels: {feat_dim}")
        
        except ImportError:
            print("Warning: ultralytics not installed. Using fallback simple backbone.")
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, 2, 1),
                nn.BatchNorm2d(32),
                nn.SiLU(inplace=True),
                nn.Conv2d(32, 64, 3, 2, 1),
                nn.BatchNorm2d(64),
                nn.SiLU(inplace=True),
                nn.Conv2d(64, 128, 3, 2, 1),
                nn.BatchNorm2d(128),
                nn.SiLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            feat_dim = 128
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.class_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        self.bbox_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 4)
        )
    
    def load_yolo(self, model_size="n", pretrained=True):
        from ultralytics import YOLO

        weights_dir = "model_weights"
        os.makedirs(weights_dir, exist_ok=True)

        model_file = f"yolov8{model_size}.pt" if pretrained else f"yolov8{model_size}.yaml"
        local_path = os.path.join(weights_dir, model_file)

        # If file already exists locally, use it
        if os.path.exists(local_path):
            model = YOLO(local_path)
            return model
        
        # Otherwise load from ultralytics (will download if needed)
        # Note: In distributed training, only rank 0 should reach this point
        # as pre-downloading is handled in train_ddp.py
        model = YOLO(model_file)

        # Move downloaded file to model_weights if it exists elsewhere
        if hasattr(model, 'ckpt_path') and os.path.exists(model.ckpt_path) and model.ckpt_path != local_path:
            try:
                shutil.move(model.ckpt_path, local_path)
            except (FileNotFoundError, shutil.Error):
                # File might have been moved by another process or doesn't exist
                pass

        return model
    
    def forward(self, x):
        feats = self.backbone(x)
        
        if len(feats.shape) == 4 and (feats.shape[2] > 1 or feats.shape[3] > 1):
            feats = self.global_pool(feats)
        
        feats = feats.view(feats.size(0), -1)
        
        logits = self.class_head(feats)
        bbox = self.bbox_head(feats)
        
        return logits, bbox


def create_yolo_nano(num_classes, pretrained=True):
    return YOLODetector(num_classes, model_size='n', pretrained=pretrained)


def create_yolo_small(num_classes, pretrained=True):
    return YOLODetector(num_classes, model_size='s', pretrained=pretrained)


def create_yolo_medium(num_classes, pretrained=True):
    return YOLODetector(num_classes, model_size='m', pretrained=pretrained)


def create_yolo_large(num_classes, pretrained=True):
    return YOLODetector(num_classes, model_size='l', pretrained=pretrained)
