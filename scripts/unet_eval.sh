#!/bin/bash -l
#SBATCH --job-name=eval_unet
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --account=p200981

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8

#SBATCH --output=unet_eval.out
#SBATCH --error=unet_eval.err

echo "=================================================="
echo "UNet Evaluation"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=================================================="

module load Python/3.11.10-GCCcore-13.3.0
module load scikit-learn/1.5.2-gfbf-2024a
module load matplotlib/3.9.2-gfbf-2024a

module load PyTorch/2.3.0-foss-2024a-CUDA-12.6.0
module load torchvision/0.18.1-foss-2024a-CUDA-12.6.0

source ds_env/bin/activate

# === DISTRIBUTED SETUP ===
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

echo "Nodes: $SLURM_NNODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Total GPUs: $(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))"
echo "Master: $MASTER_ADDR:$MASTER_PORT"

# Run UNet evaluation
srun python -u src/unet/eval_unet.py \
    --model_path model_weights_unet/unet_best \
    --val_csv /project/scratch/p200981/spark2024/val.csv \
    --image_root /project/scratch/p200981/spark2024/images \
    --mask_root /project/scratch/p200981/spark2024/mask \
    --target_size 512 \
    --batch_size 64 \
    --output_path evaluation_results_unet \
    --save_examples

echo ""
echo "=================================================="
echo "Job completed: $(date)"
echo "=================================================="
