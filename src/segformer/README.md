# SegFormer Semantic Segmentation

This directory contains a complete implementation of SegFormer for semantic segmentation on the SPARK dataset.

## Overview

**SegFormer** is a hierarchical Transformer encoder with a lightweight MLP decoder, designed for efficient semantic segmentation. It provides an excellent trade-off between accuracy and efficiency.

Reference: ["SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers"](https://arxiv.org/abs/2105.15203) - Xie et al., NeurIPS 2021

### Model Variants

| Variant | Parameters | Description |
|---------|------------|-------------|
| b0 | 3.7M | Fastest, good for prototyping |
| b1 | 13.7M | Light model |
| **b2** | **24.7M** | **Recommended - good balance** |
| b3 | 44.6M | Higher accuracy |
| b4 | 61.4M | High accuracy |
| b5 | 81.4M | Highest accuracy |

## Why SegFormer over UNet?

1. **Pretrained Backbone**: SegFormer uses ImageNet/ADE20K pretrained weights, providing better feature representations
2. **Global Context**: Transformer attention captures long-range dependencies
3. **Efficiency**: Hierarchical design is more efficient than ViT for dense prediction
4. **No Positional Encoding**: Uses Mix-FFN instead, allowing flexible input sizes
5. **Proven Results**: State-of-the-art on multiple segmentation benchmarks

## Files

- `config_segformer.yaml` - Configuration file with all hyperparameters
- `segformer_model.py` - Model wrapper with loss functions
- `train_segformer.py` - DeepSpeed distributed training script
- `eval_segformer.py` - Evaluation script with detailed metrics
- `inference_segformer.py` - Inference script for submission generation

## Quick Start

### 1. Training (Multi-Node)

```bash
# Submit training job
sbatch scripts/segformer_train.sh

# Or with custom parameters
SEGFORMER_VARIANT=b3 IMAGE_SIZE=640 sbatch scripts/segformer_train.sh
```

### 2. Evaluation

```bash
# Submit evaluation job
MODEL_PATH=model_weights_segformer_b2/segformer_best sbatch scripts/segformer_eval.sh
```

### 3. Inference

```bash
# Generate submission
MODEL_PATH=model_weights_segformer_b2/segformer_best sbatch scripts/segformer_inference.sh
```

## Configuration

All hyperparameters are controlled via `config_segformer.yaml`:

```yaml
model:
  variant: "b2"           # Model size: b0-b5
  pretrained: "ade20k"    # Pretrained weights
  freeze_backbone: false  # Freeze encoder

training:
  epochs: 100
  image_size: 512         # Input resolution
  batch_size: 8           # Per GPU
  optimizer:
    type: "adamw"
    lr: 6e-5              # Lower LR for transformers
    weight_decay: 0.01
  loss:
    type: "ce_dice"       # Combined loss
```

## Training Tips for Better Scores

### 1. Model Selection
- Start with **b2** for a good balance
- Try **b3** or **b4** if you have more GPU memory
- Use pretrained weights from **ade20k** (150 classes → transfers well)

### 2. Image Size
- Default: 512x512
- Try 640x640 or 768x768 for higher accuracy (requires smaller batch size)
- SegFormer handles various sizes well due to no positional encoding

### 3. Learning Rate
- Transformers need lower LR than CNNs
- Recommended: 2e-5 to 6e-5
- Use warmup (5-10 epochs)

### 4. Data Augmentation
- Horizontal/vertical flips
- Random rotation (up to 30°)
- Color jitter
- Consider Mosaic augmentation for small objects

### 5. Loss Function
- `ce_dice` works well for most cases
- Try `focal` loss if class imbalance is severe
- Adjust class weights if one class dominates

### 6. Multi-Node Training
- Uses DeepSpeed ZeRO-1 by default (optimizer state partitioning)
- Increase to ZeRO-2 for larger models (b4/b5)
- Effective batch size = batch_size × num_gpus

## Reproducibility

Training parameters are automatically saved to:
- `{save_path}/training_params.yaml`
- `{save_path}/training_params.json`

This includes all configuration, runtime info, and hardware details.

## Citation

```bibtex
@inproceedings{xie2021segformer,
  title={SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers},
  author={Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M and Luo, Ping},
  booktitle={NeurIPS},
  year={2021}
}
```
