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

#SBATCH --output=segformer_eval.out
#SBATCH --error=segformer_eval.err

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
module load CUDA/12.6.0
module load OpenMPI/5.0.3-GCC-13.3.0 

source ds_env/bin/activate

# Configuration
MODEL_PATH="/project/home/p200776/u103235/cvia/CVIA-25-project-team-theta/model_weights_segformer_b4_768/segformer_best"
CONFIG_FILE="/project/home/p200776/u103235/cvia/CVIA-25-project-team-theta/src/segformer/config_segformer_b4.yaml"
OUTPUT_DIR="evaluation_results_segformer_b4_768"
BATCH_SIZE="${BATCH_SIZE:-8}"
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
