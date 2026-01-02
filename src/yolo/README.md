# YOLO Object Detection for SPARK Dataset

This module implements YOLO (You Only Look Once) object detection for the SPARK spacecraft dataset.

## Features

- **10-class detection**: Detects VenusExpress, Cheops, LisaPathfinder, ObservationSat1, Proba2, Proba3, Proba3ocs, Smart1, Soho, and XMM Newton
- **Configurable via YAML**: All training parameters are set in `config_yolo.yaml`
- **Multiple model sizes**: Supports yolo11n (nano), yolo11s (small), yolo11m (medium), yolo11l (large), yolo11x (xlarge)
- **Automatic dataset conversion**: Converts SPARK CSV format to YOLO format automatically
- **Multi-GPU training**: Supports DDP via Ultralytics (up to 4 GPUs per node)
- **Augmentation presets**: None, light, medium, heavy augmentation levels

## Files

```
src/yolo/
├── config_yolo.yaml      # Training configuration
├── dataset_yolo.py       # Dataset conversion and utilities
├── yolo_model.py         # YOLO model wrapper
├── train_yolo.py         # Training script
├── eval_yolo.py          # Evaluation script
├── inference_yolo.py     # Inference script for submission
└── __init__.py           # Module exports

scripts/
├── yolo_train.sh         # SLURM training script (1 node, 4 GPUs)
├── yolo_train_multinode.sh # SLURM multi-node script (4 nodes, 16 GPUs)
├── yolo_eval.sh          # SLURM evaluation script
└── yolo_inference.sh     # SLURM inference script
```

## Quick Start

### 1. Training

Edit `src/yolo/config_yolo.yaml` to configure training:

```yaml
model:
  size: "n"          # Model size: n, s, m, l, x
  
training:
  epochs: 100
  imgsz: 640         # Image size
  batch_size: 16     # Batch size per GPU
  patience: 15       # Early stopping patience
  
augmentation:
  level: "light"     # none, light, medium, heavy
```

Submit training job:
```bash
sbatch scripts/yolo_train.sh
```

### 2. Evaluation

After training, evaluate the model:
```bash
sbatch scripts/yolo_eval.sh
```

Or manually:
```bash
python src/yolo/eval_yolo.py \
    --model_path model_weights_yolo/yolo11n_640/weights/best.pt \
    --data_yaml /project/scratch/p200981/spark2024_yolo/data.yaml
```

### 3. Inference

Generate competition submission:
```bash
sbatch scripts/yolo_inference.sh
```

Or manually:
```bash
python src/yolo/inference_yolo.py \
    --model_path model_weights_yolo/yolo11n_640/weights/best.pt \
    --data_path /path/to/test/images \
    --output_dir submission_output
```

## Configuration Options

### Model Sizes

| Size | Name        | Parameters | Speed    | Accuracy |
|------|-------------|------------|----------|----------|
| n    | Nano        | ~2.6M      | Fastest  | Good     |
| s    | Small       | ~9.4M      | Fast     | Better   |
| m    | Medium      | ~20.1M     | Medium   | Great    |
| l    | Large       | ~25.3M     | Slower   | Excellent|
| x    | XLarge      | ~56.9M     | Slowest  | Best     |

### Augmentation Levels

- **none**: No augmentation (for testing)
- **light**: Basic flips, small translations/scales (recommended for start)
- **medium**: More aggressive geometric transforms
- **heavy**: Full augmentation with MixUp, CopyPaste, RandAugment

## Output Structure

Training creates:
```
model_weights_yolo/
└── yolo11n_640/
    ├── weights/
    │   ├── best.pt      # Best model weights
    │   └── last.pt      # Last checkpoint
    ├── results.csv      # Training metrics
    └── ...              # Plots and logs
```

Inference creates:
```
submission_output/
├── detection.csv              # Detection results
└── detection_submission.zip   # Ready for submission
```

## Notes

- **Pretrained backbone**: The backbone uses COCO pretrained weights, but the detection head is randomly initialized for the 10 SPARK classes
- **Dataset conversion**: First run automatically converts SPARK CSV format to YOLO format at `/project/scratch/p200981/spark2024_yolo`
- **Multi-GPU**: Ultralytics handles DDP internally when `device=0,1,2,3` is specified
- **Early stopping**: Training stops if validation loss doesn't improve for `patience` epochs

## Comparison with DETR

| Feature           | YOLO                    | DETR                      |
|-------------------|-------------------------|---------------------------|
| Architecture      | Single-stage detector   | Transformer-based         |
| Speed             | Very fast               | Slower                    |
| Training          | Simple, stable          | Requires longer training  |
| Multi-GPU         | Ultralytics DDP         | DeepSpeed                 |
| Preprocessing     | YOLO format             | COCO format               |
