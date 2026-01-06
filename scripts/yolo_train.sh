#!/bin/bash -l
#SBATCH --job-name=train_yolo
#SBATCH --time=7:00:00
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --account=p200776

# For multi-GPU training, use 1 node with 4 GPUs
# Ultralytics handles DDP internally when device=0,1,2,3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32

#SBATCH --output=yolo_train.out
#SBATCH --error=yolo_train.err

echo "=================================================="
echo "YOLO Training - SPARK Detection Dataset"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 4"
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

# === GPU SETUP ===
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Navigate to project directory
cd ${SLURM_SUBMIT_DIR}

echo ""
echo "Configuration:"
echo "  Config file: src/yolo/config_yolo.yaml"
echo "  GPUs: 4 (DDP handled by Ultralytics)"
echo ""

echo "Starting YOLO training..."

# Run training - Ultralytics YOLO handles DDP internally when device=0,1,2,3
python -u src/yolo/train_yolo.py \
    --config src/yolo/config_yolo.yaml \
    --device 0,1,2,3

echo ""
echo "=================================================="
echo "Job completed: $(date)"
echo "=================================================="
echo "=================================================="
