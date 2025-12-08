# UNet3+ with Pretrained Backbone

This module implements **UNet3+** with pretrained backbone encoders from **timm** (PyTorch Image Models) for semantic segmentation.

## Overview

UNet3+ introduces full-scale skip connections to capture multi-scale features, combining deep semantic features with fine-grained details from all encoder levels. This implementation uses pretrained backbones (ConvNeXt, HRNet, EfficientNet, ResNet) for improved feature extraction.

## Features

- **Multiple pretrained backbones**: ConvNeXt, HRNet, EfficientNet, ResNet
- **Full-scale skip connections**: UNet3+ style multi-scale feature aggregation
- **Deep supervision**: Optional auxiliary outputs for better gradient flow
- **DeepSpeed integration**: Distributed training with ZeRO optimization
- **3-class segmentation**: Background, spacecraft body, solar panels

## Supported Backbones

| Backbone | Parameters | Recommended Use |
|----------|------------|-----------------|
| `convnext_tiny` | ~28M | **Default**, good balance of speed and accuracy |
| `convnext_small` | ~50M | Higher accuracy, more compute |
| `convnext_base` | ~89M | Best accuracy, high compute |
| `hrnet_w18` | ~9M | Lightweight, fast inference |
| `hrnet_w32` | ~29M | Good for high-resolution images |
| `hrnet_w48` | ~65M | High accuracy for detailed segmentation |
| `efficientnet_b0` | ~5M | Very lightweight |
| `efficientnet_b3` | ~12M | Good efficiency-accuracy trade-off |
| `resnet34` | ~21M | Classic, well-tested |
| `resnet50` | ~25M | Good baseline |

## Installation

Ensure you have the required dependencies:

```bash
pip install torch torchvision timm deepspeed
```

## Usage

### Training

#### Basic Training (Single GPU)
```bash
python src/unet3plus/train_unet3plus.py \
    --backbone convnext_tiny \
    --pretrained \
    --epochs 30 \
    --batch_size 8 \
    --target_size 512 \
    --save_path model_weights_unet3plus
```

#### Distributed Training with DeepSpeed (Multi-GPU)
```bash
# Submit SLURM job
sbatch scripts/unet3plus_train.sh

# Or run directly with DeepSpeed
deepspeed src/unet3plus/train_unet3plus.py \
    --backbone convnext_tiny \
    --pretrained \
    --epochs 30 \
    --batch_size 8 \
    --deepspeed
```

#### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--backbone` | `convnext_tiny` | Backbone architecture |
| `--pretrained` | `True` | Use pretrained backbone weights |
| `--no_pretrained` | - | Disable pretrained weights |
| `--freeze_backbone` | `False` | Freeze backbone during training |
| `--deep_supervision` | `False` | Enable deep supervision |
| `--cat_channels` | `64` | Channels per skip connection |
| `--epochs` | `30` | Number of training epochs |
| `--batch_size` | `8` | Batch size per GPU |
| `--target_size` | `512` | Image size for training |
| `--num_workers` | `7` | DataLoader workers |
| `--patience` | `15` | Early stopping patience |
| `--validate_every` | `2` | Validation frequency (epochs) |
| `--save_path` | `model_weights_unet3plus` | Output directory |
| `--data_root` | `/project/scratch/p200981/spark2024` | Data root |
| `--downsample` | `False` | Downsample dataset (debugging) |
| `--downsample_size` | `100` | Samples when downsampling |

### Evaluation

```bash
python src/unet3plus/eval_unet3plus.py \
    --model_path model_weights_unet3plus/unet3plus_best \
    --val_csv /project/scratch/p200981/spark2024/val.csv \
    --image_root /project/scratch/p200981/spark2024/images \
    --mask_root /project/scratch/p200981/spark2024/mask \
    --target_size 512 \
    --batch_size 32 \
    --output_path evaluation_results_unet3plus \
    --save_examples \
    --fp16
```

#### Evaluation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_path` | `model_weights_unet3plus/unet3plus_best` | Path to model |
| `--val_csv` | `...` | Validation CSV file |
| `--image_root` | `...` | Images directory |
| `--mask_root` | `...` | Masks directory |
| `--target_size` | `512` | Image size |
| `--batch_size` | `32` | Batch size |
| `--output_path` | `evaluation_results_unet3plus` | Output directory |
| `--save_examples` | `False` | Save prediction examples |
| `--fp16` | `True` | Use FP16 for faster inference |
| `--compile` | `False` | Use torch.compile (PyTorch 2.x) |

### Inference

```bash
python src/unet3plus/inference_unet3plus.py \
    --model_path model_weights_unet3plus/unet3plus_best \
    --data_path /project/scratch/p200981/spark2024_test/segmentation/stream-1-test \
    --output_dir submission_output_unet3plus \
    --batch_size 32 \
    --target_size 512 \
    --fp16
```

#### Inference Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_path` | **required** | Path to model |
| `--data_path` | **required** | Test images directory |
| `--output_dir` | `submission_output_unet3plus` | Output directory |
| `--batch_size` | `32` | Batch size |
| `--target_size` | `512` | Image size |
| `--num_workers` | `8` | DataLoader workers |
| `--fp16` | `True` | Use FP16 |
| `--no_fp16` | - | Disable FP16 |
| `--save_visualizations` | `False` | Save RGB visualizations |

## SLURM Scripts

Pre-configured SLURM scripts are available:

```bash
# Training (4 nodes, 16 GPUs total)
sbatch scripts/unet3plus_train.sh

# Evaluation (1 GPU)
sbatch scripts/unet3plus_eval.sh

# Inference (4 GPUs)
sbatch scripts/unet3plus_inference.sh
```

## Output Format

### Training Outputs
```
model_weights_unet3plus/
├── unet3plus_best/           # Best model (by validation loss)
│   ├── unet3plus_model.pt    # Model weights
│   └── config.json           # Model configuration
├── unet3plus_final/          # Final model
├── unet3plus_epoch_10/       # Checkpoint at epoch 10
└── training_plots.png        # Loss/IoU/Dice curves
```

### Evaluation Outputs
```
evaluation_results_unet3plus/
├── metrics.json              # All metrics in JSON format
├── confusion_matrix.png      # Confusion matrix visualization
├── evaluation_report.txt     # Detailed text report
└── examples/                 # Example predictions (if --save_examples)
    ├── example_0.png
    └── ...
```

### Inference Outputs
```
submission_output_unet3plus/
├── test_00000_layer.npz      # NPZ segmentation mask
├── test_00001_layer.npz
└── ...
```

## Training with Different Backbones

### ConvNeXt-Tiny (Recommended)
```bash
python src/unet3plus/train_unet3plus.py --backbone convnext_tiny --pretrained
```

### HRNet-W32 (Good for high-resolution)
```bash
python src/unet3plus/train_unet3plus.py --backbone hrnet_w32 --pretrained
```

### EfficientNet-B0 (Lightweight)
```bash
python src/unet3plus/train_unet3plus.py --backbone efficientnet_b0 --pretrained
```

### Frozen Backbone (Fine-tune decoder only)
```bash
python src/unet3plus/train_unet3plus.py --backbone convnext_tiny --pretrained --freeze_backbone
```

### Deep Supervision
```bash
python src/unet3plus/train_unet3plus.py --backbone convnext_tiny --pretrained --deep_supervision
```

## Model Architecture

```
Input Image (3, H, W)
       │
       ▼
┌──────────────────────────────────────┐
│     Pretrained Backbone (timm)       │
│  (ConvNeXt/HRNet/EfficientNet/ResNet)│
└──────────────────────────────────────┘
       │
       ▼ (Multi-scale features: E1, E2, E3, E4)
       │
┌──────────────────────────────────────┐
│        UNet3+ Decoder                │
│  (Full-scale skip connections)       │
│                                      │
│  D4 ← E4                             │
│  D3 ← E3 + upsample(D4)              │
│  D2 ← E2 + upsample(D3)              │
│  D1 ← E1 + upsample(D2)              │
│                                      │
│  Multi-scale Aggregation:            │
│  [D1, D2↑, D3↑↑, D4↑↑↑] → Fusion     │
└──────────────────────────────────────┘
       │
       ▼
   Output (3, H, W)
   (3-class segmentation)
```

## Classes

- **Class 0**: Background (black)
- **Class 1**: Spacecraft body (red)
- **Class 2**: Solar panels (blue)

## References

- UNet3+: [A Full-Scale Connected UNet for Medical Image Segmentation](https://arxiv.org/abs/2004.08790)
- timm: [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models)
- ConvNeXt: [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
- HRNet: [Deep High-Resolution Representation Learning](https://arxiv.org/abs/1908.07919)
