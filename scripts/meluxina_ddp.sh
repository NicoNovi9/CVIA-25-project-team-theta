#!/bin/bash -l
#SBATCH --job-name=train_ddp
#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --account=p200981

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4

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

export CUDA_VISIBLE_DEVICES=0,1,2,3

srun python src/train_ddp.py \
    --deepspeed --deepspeed_config ds_config.json

# echo "Starting DDP training script on 4 GPUs..."
# torchrun \
#     --nnodes=1 \
#     --nproc_per_node=4 \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint=localhost:0 \
#     src/train_ddp.py
