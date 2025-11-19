import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
from torch.utils.data import Subset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os

from models.simple_model import SimpleDetector
from utils.spark_detection_dataset import SparkDetectionDataset


def setup_ddp():
    """Initialize the distributed environment."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    """Clean up the distributed environment."""
    dist.destroy_process_group()


if __name__ == "__main__":
    DATA_ROOT = "/project/scratch/p200981/spark2024"
    DOWN_SAMPLE = False
    DOWN_SAMPLE_SUBSET = 10
    BATCH_SIZE = 2056
    N_EPOCHS = 100
    LEARNING_RATE = 1e-3

    # Initialize DDP
    local_rank = setup_ddp()
    world_size = dist.get_world_size()
    is_main_process = local_rank == 0

    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])

    if is_main_process:
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

    if is_main_process:
        print(f"Data prepared: train samples = {len(train_dataset)}, val samples = {len(val_dataset)}")

    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True
    )

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=False
    )

    if is_main_process:
        print("Preparing DataLoaders")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=8,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        num_workers=8,
        pin_memory=True
    )

    device = f"cuda:{local_rank}"

    model = SimpleDetector(num_classes=10).to(device)
    model = DDP(model, device_ids=[local_rank])

    ce_loss = nn.CrossEntropyLoss()
    bbox_loss_fn = nn.SmoothL1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if is_main_process:
        print(f"Using {world_size} GPUs for training")
        print("Beginning training pipeline")

    for epoch in range(N_EPOCHS):
        if is_main_process:
            print(f"Starting epoch {epoch+1:03d}")
        
        # Set epoch for sampler to ensure proper shuffling
        train_sampler.set_epoch(epoch)
        
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

        # Gather losses from all processes
        total_loss_tensor = torch.tensor([total_loss], device=device)
        total_cls_loss_tensor = torch.tensor([total_cls_loss], device=device)
        total_bbox_loss_tensor = torch.tensor([total_bbox_loss], device=device)
        num_batches_tensor = torch.tensor([num_batches], device=device)

        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_cls_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bbox_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)

        avg_loss = total_loss_tensor.item() / num_batches_tensor.item()
        avg_cls_loss = total_cls_loss_tensor.item() / num_batches_tensor.item()
        avg_bbox_loss = total_bbox_loss_tensor.item() / num_batches_tensor.item()

        if is_main_process:
            print(f"TRAIN {epoch+1:03d} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Cls: {avg_cls_loss:.4f} | "
                  f"BBox: {avg_bbox_loss:.4f}")

        # VALIDATION
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

        # Gather validation losses from all processes
        val_loss_tensor = torch.tensor([val_loss], device=device)
        val_cls_loss_tensor = torch.tensor([val_cls_loss], device=device)
        val_bbox_loss_tensor = torch.tensor([val_bbox_loss], device=device)
        val_batches_tensor = torch.tensor([val_batches], device=device)

        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_cls_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_bbox_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_batches_tensor, op=dist.ReduceOp.SUM)

        avg_val_loss = val_loss_tensor.item() / val_batches_tensor.item()
        avg_val_cls = val_cls_loss_tensor.item() / val_batches_tensor.item()
        avg_val_bbox = val_bbox_loss_tensor.item() / val_batches_tensor.item()

        if is_main_process:
            print(f"VAL   {epoch+1:03d} | "
                  f"Loss: {avg_val_loss:.4f} | "
                  f"Cls: {avg_val_cls:.4f} | "
                  f"BBox: {avg_val_bbox:.4f}")
    
    # Only save model from main process
    if is_main_process:
        torch.save(model.module.state_dict(), "model_weights/simple_model_weights.pth")
        print("Saving model to model_weights/simple_model_weights.pth")

    cleanup_ddp()
