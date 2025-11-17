import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torch.utils.data import Subset


from simple_model import SimpleDetector
from spark_detection_dataset import SparkDetectionDataset


if __name__ == "__main__":
    DATA_ROOT = "/project/scratch/p200981/spark2024"

    print("Preparing dataset...", flush=True)

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

    train_dataset = Subset(train_dataset, range(500))
    loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)

    print("Initializing model...")

    model = SimpleDetector(num_classes=10)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    ce_loss = nn.CrossEntropyLoss()
    bbox_loss_fn = nn.SmoothL1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Starting training...")

    for epoch in range(50):
        print(f"Starting epoch {epoch+1:03d}")

        total_loss = 0.0
        total_cls_loss = 0.0
        total_bbox_loss = 0.0
        num_batches = 0

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

            total_loss += loss.item()
            total_cls_loss += loss_cls.item()
            total_bbox_loss += loss_bbox.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_cls_loss = total_cls_loss / num_batches
        avg_bbox_loss = total_bbox_loss / num_batches

        print(f"Epoch {epoch+1:03d} | "
            f"Loss: {avg_loss:.4f} | "
            f"Cls: {avg_cls_loss:.4f} | "
            f"BBox: {avg_bbox_loss:.4f}")

