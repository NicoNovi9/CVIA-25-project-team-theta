#!/bin/bash -l
#SBATCH --job-name=infer_segformer
#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --account=p200981

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4

#SBATCH --output=segformer_inference.out
#SBATCH --error=segformer_inference.err

echo "=================================================="
echo "SegFormer Inference"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Start time: $(date)"
echo "=================================================="

# Module loading
module purge
module load Python/3.11.10-GCCcore-13.3.0
module load CUDA/12.6.0
module load OpenMPI/5.0.3-GCC-13.3.0 

source ds_env/bin/activate

export CUDA_VISIBLE_DEVICES=0,1,2,3

# Configuration
MODEL_PATH="/project/home/p200776/u103235/cvia/CVIA-25-project-team-theta/model_weights_segformer_b4_768/segformer_best"
DATA_PATH="${DATA_PATH:-/project/scratch/p200981/spark2024_test/segmentation/stream-1-test}"
OUTPUT_DIR="submission_segformer_b4_768"
BATCH_SIZE="${BATCH_SIZE:-8}"
IMAGE_SIZE="${IMAGE_SIZE:-768}"
echo ""
echo "Inference Configuration:"
echo "  Model path: $MODEL_PATH"
echo "  Data path: $DATA_PATH"
echo "  Output dir: $OUTPUT_DIR"
echo "  Batch size: $BATCH_SIZE"
echo "  Image size: $IMAGE_SIZE"
echo ""

# Run inference
python -u src/segformer/inference_segformer.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --image_size "$IMAGE_SIZE" \
    --tta

echo ""
echo "=================================================="
echo "Inference completed: $(date)"
echo "=================================================="
