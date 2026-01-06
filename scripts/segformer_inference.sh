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

#SBATCH --output=segformer_inference_%j.out
#SBATCH --error=segformer_inference_%j.err

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
module load scikit-learn/1.5.2-gfbf-2024a
module load matplotlib/3.9.2-gfbf-2024a
module load PyTorch/2.3.0-foss-2024a-CUDA-12.6.0
module load torchvision/0.18.1-foss-2024a-CUDA-12.6.0

source ds_env/bin/activate

export CUDA_VISIBLE_DEVICES=0,1,2,3

# Configuration
MODEL_PATH="${MODEL_PATH:-model_weights_segformer_b2/segformer_best}"
DATA_PATH="${DATA_PATH:-/project/scratch/p200981/spark2024_test/segmentation/stream-1-test}"
OUTPUT_DIR="${OUTPUT_DIR:-submission_segformer}"
BATCH_SIZE="${BATCH_SIZE:-64}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"

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
    --fp16 \
    --tta

# Create submission zip if requested
if [ "${CREATE_ZIP:-true}" = "true" ]; then
    echo ""
    echo "Creating submission zip..."
    cd "$OUTPUT_DIR"
    zip -r ../submission_segformer.zip *.npz
    cd ..
    echo "Created submission_segformer.zip"
fi

echo ""
echo "=================================================="
echo "Inference completed: $(date)"
echo "=================================================="
