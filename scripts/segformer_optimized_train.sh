#!/bin/bash -l
#SBATCH --job-name=train_segformer_opt
#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --account=p200981

# Multi-node configuration
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4

#SBATCH --output=segformer_optimized_train.out
#SBATCH --error=segformer_optimized_train.err

echo "=================================================="
echo "SegFormer Optimized (YOLO-guided) Multi-Node Training"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
echo "Total GPUs: $SLURM_NTASKS"
echo "Node list: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=================================================="

# =============================================================================
# MODULE LOADING
# =============================================================================
module purge
module load Python/3.11.10-GCCcore-13.3.0
module load CUDA/12.6.0
module load OpenMPI/5.0.3-GCC-13.3.0 

# =============================================================================
# VIRTUAL ENVIRONMENT SETUP
# =============================================================================
source ds_env/bin/activate

# Verify dependencies
python - << 'EOF'
import torch
import transformers
import torchvision

print("Python       :", torch.sys.version.split()[0])
print("Torch        :", torch.__version__)
print("Torch CUDA   :", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count    :", torch.cuda.device_count())
print("Transformers :", transformers.__version__)
print("Torchvision  :", torchvision.__version__)

# Check ultralytics for YOLO
try:
    from ultralytics import YOLO
    print("Ultralytics  : installed")
except ImportError:
    print("Ultralytics  : NOT INSTALLED - run: pip install ultralytics")
EOF

# =============================================================================
# CUDA SETUP
# =============================================================================
GPUS_PER_NODE=$((SLURM_NTASKS_PER_NODE * 1))
if [ "$GPUS_PER_NODE" -eq 1 ]; then
    export CUDA_VISIBLE_DEVICES=0
elif [ "$GPUS_PER_NODE" -eq 2 ]; then
    export CUDA_VISIBLE_DEVICES=0,1
elif [ "$GPUS_PER_NODE" -eq 3 ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2
else
    export CUDA_VISIBLE_DEVICES=0,1,2,3
fi

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
CONFIG_FILE="${CONFIG_FILE:-src/segformer_optimized/config_segformer_optimized.yaml}"

# Optional CLI overrides
VARIANT="${VARIANT:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
EPOCHS="${EPOCHS:-}"
LR="${LR:-}"
YOLO_MODEL="${YOLO_MODEL:-}"
SAVE_PATH="${SAVE_PATH:-}"

echo ""
echo "Training Configuration:"
echo "  Config file: $CONFIG_FILE"
if [ -n "$VARIANT" ]; then echo "  Variant override: $VARIANT"; fi
if [ -n "$BATCH_SIZE" ]; then echo "  Batch size override: $BATCH_SIZE"; fi
if [ -n "$EPOCHS" ]; then echo "  Epochs override: $EPOCHS"; fi
if [ -n "$LR" ]; then echo "  Learning rate override: $LR"; fi
if [ -n "$YOLO_MODEL" ]; then echo "  YOLO model override: $YOLO_MODEL"; fi
if [ -n "$SAVE_PATH" ]; then echo "  Save path override: $SAVE_PATH"; fi
echo ""

# =============================================================================
# BUILD TRAINING COMMAND
# =============================================================================
TRAIN_CMD="python -u src/segformer_optimized/train_segformer_optimized.py --config $CONFIG_FILE"

if [ -n "$VARIANT" ]; then TRAIN_CMD="$TRAIN_CMD --variant $VARIANT"; fi
if [ -n "$BATCH_SIZE" ]; then TRAIN_CMD="$TRAIN_CMD --batch_size $BATCH_SIZE"; fi
if [ -n "$EPOCHS" ]; then TRAIN_CMD="$TRAIN_CMD --epochs $EPOCHS"; fi
if [ -n "$LR" ]; then TRAIN_CMD="$TRAIN_CMD --lr $LR"; fi
if [ -n "$YOLO_MODEL" ]; then TRAIN_CMD="$TRAIN_CMD --yolo_model $YOLO_MODEL"; fi
if [ -n "$SAVE_PATH" ]; then TRAIN_CMD="$TRAIN_CMD --save_path $SAVE_PATH"; fi

echo "Running: srun $TRAIN_CMD"
echo ""

# =============================================================================
# RUN TRAINING
# =============================================================================
srun $TRAIN_CMD

echo ""
echo "=================================================="
echo "Training completed: $(date)"
echo "=================================================="
