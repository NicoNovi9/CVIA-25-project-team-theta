#!/bin/bash -l
#SBATCH --job-name=infer_unet3plus
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --account=p200981

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4

#SBATCH --output=unet3plus_inference.out
#SBATCH --error=unet3plus_inference.err

echo "=================================================="
echo "UNet3+ Inference"
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

export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "Running inference..."

python src/unet3plus/inference_unet3plus.py \
    --model_path /project/home/p200776/u103235/cvia/local/models_trained/3_model_weights_unet3plus_30epochs/unet3plus_best \
    --data_path /project/scratch/p200981/spark2024_test/segmentation/stream-1-test \
    --output_dir submission_output_unet3plus \
    --batch_size 8 \
    --target_size 512 \
    --num_workers 8 \
    --fp16

echo ""
echo "=================================================="
echo "Inference completed: $(date)"
