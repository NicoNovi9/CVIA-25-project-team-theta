#!/bin/bash -l
#SBATCH --job-name=eval_segformer
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --account=p200981

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

#SBATCH --output=segformer_eval_%j.out
#SBATCH --error=segformer_eval_%j.err

echo "=================================================="
echo "SegFormer Evaluation"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
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

# Configuration
MODEL_PATH="${MODEL_PATH:-model_weights_segformer_b2/segformer_best}"
CONFIG_FILE="${CONFIG_FILE:-src/segformer/config_segformer.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluation_results_segformer}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SPLIT="${SPLIT:-val}"

echo ""
echo "Evaluation Configuration:"
echo "  Model path: $MODEL_PATH"
echo "  Config file: $CONFIG_FILE"
echo "  Output dir: $OUTPUT_DIR"
echo "  Batch size: $BATCH_SIZE"
echo "  Split: $SPLIT"
echo ""

# Run evaluation
python -u src/segformer/eval_segformer.py \
    --model_path "$MODEL_PATH" \
    --config "$CONFIG_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --split "$SPLIT" \
    --save_examples

echo ""
echo "=================================================="
echo "Evaluation completed: $(date)"
echo "=================================================="
