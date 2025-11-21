import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
from torch.utils.data import Subset

from models.simple_model import SimpleDetector
from models.yolo_model import YOLODetector
from utils.spark_detection_dataset import SparkDetectionDataset
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

if __name__ == "__main__":
    DATA_ROOT = "/project/scratch/p200981/spark2024"
    DOWN_SAMPLE = False
    DOWN_SAMPLE_SUBSET = 10
    BATCH_SIZE = 2056
    N_EPOCHS = 100
    LEARNING_RATE = 1e-3
    VALIDATION = True

    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])

    print("Preparing datasets")

    train_dataset = SparkDetectionDataset(
        csv_path=f"{DATA_ROOT}/train.csv",
        image_root=f"{DATA_ROOT}/images",
        split="train",
        transform=transform
    )

    val_dataset = SparkDetectionDataset(
        csv_path=f"{DATA_ROOT}/val.csv",
        image_root=f"{DATA_ROOT}/images",
        split="val",
        transform=transform
    )

    if DOWN_SAMPLE:
        train_dataset = Subset(train_dataset, range(DOWN_SAMPLE_SUBSET))
        val_dataset = Subset(val_dataset, range(DOWN_SAMPLE_SUBSET // 10))

    print(f"Data prepared: train samples = {len(train_dataset)}, val samples = {len(val_dataset)}")

    print("Preparing DataLoaders")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model = SimpleDetector(num_classes=10).to(device)
    model = YOLODetector(num_classes=10, model_size='n', pretrained=True).to(device)

    ce_loss = nn.CrossEntropyLoss()
    bbox_loss_fn = nn.SmoothL1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Using device: {device}")
    print("Beginning training pipeline")

    for epoch in range(N_EPOCHS):
        print(f"Starting epoch {epoch+1:03d}")
        model.train()

        total_loss = 0.0
        total_cls_loss = 0.0
        total_bbox_loss = 0.0
        num_batches = 0

        for imgs, bboxes, labels in train_loader:
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

        print(f"TRAIN {epoch+1:03d} | "
              f"Loss: {avg_loss:.4f} | "
              f"Cls: {avg_cls_loss:.4f} | "
              f"BBox: {avg_bbox_loss:.4f}")

        # VALIDATION
        if not VALIDATION:
            continue
            
        model.eval()

        val_loss = 0.0
        val_cls_loss = 0.0
        val_bbox_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for imgs, bboxes, labels in val_loader:
                imgs = imgs.to(device)
                bboxes = bboxes.to(device)
                labels = labels.to(device)

                logits, pred_bbox = model(imgs)

                loss_cls = ce_loss(logits, labels)
                loss_bbox = bbox_loss_fn(pred_bbox, bboxes)
                loss = loss_cls + loss_bbox

                val_loss += loss.item()
                val_cls_loss += loss_cls.item()
                val_bbox_loss += loss_bbox.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches
        avg_val_cls = val_cls_loss / val_batches
        avg_val_bbox = val_bbox_loss / val_batches

        print(f"VAL   {epoch+1:03d} | "
              f"Loss: {avg_val_loss:.4f} | "
              f"Cls: {avg_val_cls:.4f} | "
              f"BBox: {avg_val_bbox:.4f}")
    
    torch.save(model.state_dict(), "model_wights/simple_model_weights.pth")
    print("Saving model to model_wights/simple_model_weights.pth")

