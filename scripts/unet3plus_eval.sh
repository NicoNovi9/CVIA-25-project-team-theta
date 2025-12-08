#!/bin/bash -l
#SBATCH --job-name=eval_unet3plus
#SBATCH --time=0:30:00
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --account=p200981

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

#SBATCH --output=unet3plus_eval.out
#SBATCH --error=unet3plus_eval.err

echo "=================================================="
echo "UNet3+ Evaluation"
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

source ds_env/bin/activate

export CUDA_VISIBLE_DEVICES=0

echo "Running evaluation..."

python src/unet3plus/eval_unet3plus.py \
    --model_path model_weights_unet3plus/unet3plus_best \
    --val_csv /project/scratch/p200981/spark2024/val.csv \
    --image_root /project/scratch/p200981/spark2024/images \
    --mask_root /project/scratch/p200981/spark2024/mask \
    --target_size 512 \
    --batch_size 8 \
    --output_path evaluation_results_unet3plus \
    --save_examples \
    --fp16

echo ""
echo "=================================================="
echo "Evaluation completed: $(date)"
