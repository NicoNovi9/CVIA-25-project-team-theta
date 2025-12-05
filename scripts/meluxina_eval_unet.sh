#!/bin/bash -l
#SBATCH --job-name=eval_unet
#SBATCH --time=0:15:00
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --account=p200981

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1

#SBATCH --output=meluxina_eval_unet.out
#SBATCH --error=meluxina_eval_unet.err

module load Python/3.11.10-GCCcore-13.3.0
module load scikit-learn/1.5.2-gfbf-2024a
module load matplotlib/3.9.2-gfbf-2024a
module load Seaborn/0.13.2-gfbf-2024a

module load PyTorch/2.3.0-foss-2024a-CUDA-12.6.0
module load torchvision/0.18.1-foss-2024a-CUDA-12.6.0

source ds_env/bin/activate

export CUDA_VISIBLE_DEVICES=0

echo "Evaluating UNet Segmentation model (optimized)..."
python -u src/eval_unet.py \
    --model_path model_weights_unet_30epochs/unet_best \
    --val_csv /project/scratch/p200981/spark2024/val.csv \
    --image_root /project/scratch/p200981/spark2024/images \
    --mask_root /project/scratch/p200981/spark2024/mask \
    --target_size 512 \
    --batch_size 64 \
    --output_path evaluation_results_unet \
    --save_examples
