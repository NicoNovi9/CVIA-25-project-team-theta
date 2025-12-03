#!/bin/bash -l
#SBATCH --job-name=train_ddp
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu
#SBATCH --time=02:00:00 
#SBATCH --qos=default 
#SBATCH --account=p200981 
#SBATCH --output=meluxina_train.out
#SBATCH --error=meluxina_train.err

module load Python
module load CUDA/12.6.0
if [ ! -d "ds_env" ]; then
    python -m venv ds_env

    source ds_env/bin/activate

    pip install --upgrade pip
    pip install torch torchvision numpy pandas ultralytics
    pip install deepspeed mpi4py
fi

source ds_env/bin/activate

echo "Starting DDP training script on 4 GPUs..."
torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    src/train_ddp.py
