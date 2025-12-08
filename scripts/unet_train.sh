#!/bin/bash -l
#SBATCH --job-name=train_unet
#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --account=p200981

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4

#SBATCH --output=unet_train.out
#SBATCH --error=unet_train.err

echo "=================================================="
echo "UNet Training"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=================================================="

module load Python/3.11.10-GCCcore-13.3.0
module load scikit-learn/1.5.2-gfbf-2024a
module load matplotlib/3.9.2-gfbf-2024a
module load Seaborn/0.13.2-gfbf-2024a

module load PyTorch/2.3.0-foss-2024a-CUDA-12.6.0
module load torchvision/0.18.1-foss-2024a-CUDA-12.6.0

if [ ! -d "ds_env" ]; then
    python -m venv ds_env

    source ds_env/bin/activate

    pip install --upgrade pip
    pip install setuptools deepspeed mpi4py
    pip install transformers accelerate timm
    pip install pandas pycocotools
fi

source ds_env/bin/activate

export CUDA_VISIBLE_DEVICES=0,1,2,3

NODES=$SLURM_JOB_NUM_NODES
TASKS_PER_NODE=$SLURM_NTASKS_PER_NODE
GPUS_PER_TASK=1
GPUS_PER_NODE=$((TASKS_PER_NODE * GPUS_PER_TASK))
TOTAL_GPUS=$SLURM_NTASKS

echo "Starting DeepSpeed training on ${NODES} node(s), ${GPUS_PER_NODE} GPUs per node (${TOTAL_GPUS} GPUs total)..."

srun python -u src/unet/train_unet.py --deepspeed

echo ""
echo "=================================================="
echo "Job completed: $(date)"
echo "=================================================="
