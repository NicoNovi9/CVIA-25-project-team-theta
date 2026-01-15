#!/bin/bash -l
#SBATCH --job-name=train_segformer
#SBATCH --time=25:00:00
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --account=p200981

# Multi-node configuration - adjust as needed
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4

#SBATCH --output=segformer_train_b4.out
#SBATCH --error=segformer_train_b4.err

echo "=================================================="
echo "SegFormer Multi-Node Training"
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
# if [ ! -d "ds_env" ]; then
#     echo "Creating virtual environment..."
#     python -m venv ds_env
#     source ds_env/bin/activate
    
#     pip install --upgrade pip
#     pip install setuptools deepspeed mpi4py
#     pip install transformers accelerate timm
#     pip install pandas pycocotools pyyaml
# else
    source ds_env/bin/activate
# fi

# Verify transformers is installed (needed for SegFormer)
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
# All configuration is controlled via the YAML config file
# You can override specific parameters via CLI arguments if needed

# Config file path (can be overridden)
# CONFIG_FILE="${CONFIG_FILE:-src/segformer/config_segformer.yaml}"
CONFIG_FILE="src/segformer/config_segformer_b4.yaml"

echo ""
echo "Training Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  (All hyperparameters are loaded from the config file)"
echo ""

# =============================================================================
# DISTRIBUTED SETUP
# =============================================================================
NODES=$SLURM_JOB_NUM_NODES
TASKS_PER_NODE=$SLURM_NTASKS_PER_NODE
TOTAL_GPUS=$SLURM_NTASKS

echo "Starting DeepSpeed training on ${NODES} node(s), ${TASKS_PER_NODE} GPUs per node (${TOTAL_GPUS} GPUs total)..."
echo ""

# =============================================================================
# RUN TRAINING
# =============================================================================
# Use only config file, optionally override via CLI arguments if needed
srun python -u src/segformer/train_segformer.py \
    --config "$CONFIG_FILE"

# To override specific parameters, add CLI arguments:
# srun python -u src/segformer/train_segformer.py \
#     --config "$CONFIG_FILE" \
#     --variant b3 \
#     --image_size 640 \
#     --batch_size 4

# =============================================================================
# CLEANUP
# =============================================================================
echo ""
echo "=================================================="
echo "Job completed: $(date)"
echo "=================================================="
