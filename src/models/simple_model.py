import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDetector(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )

        self.class_head = nn.Linear(32, num_classes)

        self.bbox_head = nn.Linear(32, 4)

    def forward(self, x):
        # x: (B, 3, H, W)
        feats = self.features(x)     
        feats = feats.view(x.size(0), -1)  

        logits = self.class_head(feats)  
        bbox = self.bbox_head(feats)     

        return logits, bbox

if __name__ == "__main__":
    model = SimpleDetector(num_classes=10)
    x = torch.randn(4, 3, 1024, 1024)
    logits, bbox = model(x)
    print("Logits:", logits.shape)
    print("BBox:", bbox.shape)
