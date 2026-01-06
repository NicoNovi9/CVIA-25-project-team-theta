#!/bin/bash -l
#SBATCH --job-name=inference_yolo
#SBATCH --output=yolo_inference.out
#SBATCH --error=yolo_inference.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --time=01:00:00
#SBATCH --account=p200981

echo "=================================================="
echo "YOLO Detection Inference"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=================================================="

# Load modules
module load Python/3.11.10-GCCcore-13.3.0
module load scikit-learn/1.5.2-gfbf-2024a
module load matplotlib/3.9.2-gfbf-2024a

module load PyTorch/2.3.0-foss-2024a-CUDA-12.6.0
module load torchvision/0.18.1-foss-2024a-CUDA-12.6.0

source ds_env/bin/activate

# Install dependencies if needed
pip install ultralytics --quiet 2>/dev/null

export CUDA_VISIBLE_DEVICES=0,1,2,3

cd ${SLURM_SUBMIT_DIR}

# === CONFIGURATION ===
# Change these paths as needed
MODEL_PATH="model_weights_yolo/yolo11n_640/weights/best.pt"
DATA_PATH="/project/scratch/p200981/spark2024_test/detection/images"
OUTPUT_DIR="submission_output_yolo"

echo ""
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Data: $DATA_PATH"
echo "  Output directory: $OUTPUT_DIR"
echo "  GPU: 0"
echo ""

# Run YOLO detection inference
python -u src/yolo/inference_yolo.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --threshold 0.25 \
    --imgsz 640 \
    --batch_size 64 \
    --device 0

echo ""
echo "=================================================="
echo "Job completed: $(date)"
echo "=================================================="
echo ""
echo "Output files in: $OUTPUT_DIR"
echo "Submission zip: $OUTPUT_DIR/detection_submission.zip"
