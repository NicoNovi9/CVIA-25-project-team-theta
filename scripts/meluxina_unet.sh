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

#SBATCH --output=meluxina_unet.out
#SBATCH --error=meluxina_unet.err

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

echo "Starting DeepSpeed UNet Segmentation training on 1 nodes, 4 GPUs per node (4 in total)..."
srun python -u src/train_unet_ddp.py \
    --deepspeed --deepspeed_config scripts/ds_config_unet.json
