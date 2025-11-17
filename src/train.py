import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T

from simple_model import SimpleDetector
from spark_detection_dataset import SparkDetectionDataset


if __name__ == "__main__":
    DATA_ROOT = "/project/scratch/p200981/spark2024"

    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])

    train_dataset = SparkDetectionDataset(
        csv_path=f"{DATA_ROOT}/train.csv",
        image_root=f"{DATA_ROOT}/images",
        split="train",
        transform=transform
    )

    loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)

    model = SimpleDetector(num_classes=10)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    ce_loss = nn.CrossEntropyLoss()
    bbox_loss_fn = nn.SmoothL1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    for epoch in range(50):
        for imgs, bboxes, labels in loader:

            imgs = imgs.to(device)
            bboxes = bboxes.to(device)
            labels = labels.to(device)

            logits, pred_bbox = model(imgs)

            loss_cls = ce_loss(logits, labels)
            loss_bbox = bbox_loss_fn(pred_bbox, bboxes)

            loss = loss_cls + loss_bbox

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Loss: {loss.item():.4f} | cls: {loss_cls.item():.4f} | bbox: {loss_bbox.item():.4f}")
        print(f"Epoch {epoch} done.")
