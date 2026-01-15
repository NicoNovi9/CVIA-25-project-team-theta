#!/bin/bash -l
#SBATCH --job-name=infer_segformer_opt
#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --account=p200981

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4

#SBATCH --output=segformer_optimized_inference.out
#SBATCH --error=segformer_optimized_inference.err

echo "=================================================="
echo "SegFormer Optimized (YOLO-guided) Inference"
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
MODEL_PATH="/project/home/p200776/u103235/cvia/CVIA-25-project-team-theta/trained_models/segformer/11_segformer_b2_optimized512/segformer_optimized_best"
YOLO_MODEL="/project/home/p200776/u103235/cvia/CVIA-25-project-team-theta/trained_models/model_weights_yolo/7_yolo11s_640/weights/best.pt"
DATA_PATH="${DATA_PATH:-/project/scratch/p200981/spark2024_test/segmentation/stream-1-test}"
OUTPUT_DIR="${OUTPUT_DIR:-submission_segformer_optimized}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEGFORMER_SIZE="${SEGFORMER_SIZE:-512}"
BBOX_EXPANSION="${BBOX_EXPANSION:-1.1}"

echo ""
echo "Inference Configuration:"
echo "  SegFormer model: $MODEL_PATH"
echo "  YOLO model: $YOLO_MODEL"
echo "  Data path: $DATA_PATH"
echo "  Output dir: $OUTPUT_DIR"
echo "  Batch size: $BATCH_SIZE"
echo "  SegFormer size: $SEGFORMER_SIZE"
echo "  BBox expansion: $BBOX_EXPANSION"
echo ""

# Run inference
python -u src/segformer_optimized/inference_segformer_optimized.py \
    --model_path "$MODEL_PATH" \
    --yolo_model "$YOLO_MODEL" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --segformer_size "$SEGFORMER_SIZE" \
    --bbox_expansion "$BBOX_EXPANSION" \
    --tta

echo ""
echo "=================================================="
echo "Inference completed: $(date)"
echo "=================================================="

# Create submission zip
echo ""
echo "Creating submission zip..."
cd "$OUTPUT_DIR"
zip -q submission.zip *.npz
echo "Submission zip created: $OUTPUT_DIR/submission.zip"
