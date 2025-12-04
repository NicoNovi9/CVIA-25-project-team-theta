#!/bin/bash -l
#SBATCH --job-name=eval_detr
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --account=p200981

# === DISTRIBUTED CONFIGURATION ===
# Single node with 4 GPUs (change --nodes for multi-node)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8

#SBATCH --output=meluxina_eval_detr.out
#SBATCH --error=meluxina_eval_detr.err

module load Python/3.11.10-GCCcore-13.3.0
module load scikit-learn/1.5.2-gfbf-2024a
module load matplotlib/3.9.2-gfbf-2024a

module load PyTorch/2.3.0-foss-2024a-CUDA-12.6.0
module load torchvision/0.18.1-foss-2024a-CUDA-12.6.0

source ds_env/bin/activate

# Install additional dependencies if needed
pip install pycocotools --quiet

# === DISTRIBUTED SETUP ===
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

echo "======================================================"
echo "DETR DISTRIBUTED EVALUATION"
echo "======================================================"
echo "Nodes: $SLURM_NNODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Total GPUs: $(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "======================================================"

# Run with srun for multi-GPU/multi-node
srun python -u src/eval_detr.py
