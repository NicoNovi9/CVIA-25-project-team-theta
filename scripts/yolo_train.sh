#!/bin/bash -l
#SBATCH --job-name=train_yolo
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --account=p200776

# For multi-GPU training, use 1 node with 4 GPUs
# Ultralytics handles DDP internally when device=0,1,2,3

echo "=================================================="
echo "YOLO Training - SPARK Detection Dataset"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 4"
echo "Start time: $(date)"
echo "=================================================="

# Load modules
module purge
module load Python/3.11.10-GCCcore-13.3.0
module load CUDA/12.6.0
module load OpenMPI/5.0.3-GCC-13.3.0 

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
