#!/bin/bash -l
#SBATCH --job-name=train_yolo_multinode
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --account=p200981

# Multi-node training: 4 nodes x 4 GPUs = 16 GPUs total
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4

#SBATCH --output=yolo_train_multinode.out
#SBATCH --error=yolo_train_multinode.err

echo "=================================================="
echo "YOLO Multi-Node Training - SPARK Detection Dataset"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "Number of nodes: $SLURM_NNODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Total GPUs: $(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))"
echo "Start time: $(date)"
echo "=================================================="

# Load modules
module load Python/3.11.10-GCCcore-13.3.0
module load scikit-learn/1.5.2-gfbf-2024a
module load matplotlib/3.9.2-gfbf-2024a
module load Seaborn/0.13.2-gfbf-2024a

module load PyTorch/2.3.0-foss-2024a-CUDA-12.6.0
module load torchvision/0.18.1-foss-2024a-CUDA-12.6.0

# Create virtual environment if needed
if [ ! -d "ds_env" ]; then
    echo "Creating virtual environment..."
    python -m venv ds_env

    source ds_env/bin/activate

    pip install --upgrade pip
    pip install setuptools
    pip install ultralytics
    pip install pandas pyyaml tqdm
fi

source ds_env/bin/activate

# Install ultralytics if not present
pip install ultralytics --quiet 2>/dev/null

# === DISTRIBUTED SETUP ===
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Master address for distributed training
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# Calculate world size
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

echo ""
echo "Distributed Configuration:"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  WORLD_SIZE: $WORLD_SIZE"
echo ""

# Navigate to project directory
cd ${SLURM_SUBMIT_DIR}

echo "Starting YOLO multi-node training..."
echo "Configuration: src/yolo/config_yolo.yaml"
echo ""

# Multi-node training with SLURM:
# - srun launches one Python process per task (per GPU)
# - Each process gets a unique GPU via SLURM_LOCALID
# - Training script detects SLURM environment and uses single GPU per task
# - Each task trains independently with unique save directories to avoid conflicts
# - Combine checkpoints manually after training or use rank 0 results

srun python -u src/yolo/train_yolo.py \
    --config src/yolo/config_yolo.yaml

echo ""
echo "=================================================="
echo "Job completed: $(date)"
echo "=================================================="
