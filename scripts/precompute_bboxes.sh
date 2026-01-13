#!/bin/bash -l
#SBATCH --job-name=precompute_bbox
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --account=p200981
#SBATCH --qos=default

#==============================================================================
# YOLO Bounding Box Precomputation for SegFormer Optimized
#==============================================================================
# This script precomputes bounding boxes for the training split using YOLO,
# which eliminates the need for on-the-fly detection during distributed training.
#
# Usage:
#   sbatch scripts/precompute_bboxes.sh
#
# Override defaults via environment variables:
#   YOLO_MODEL=/path/to/model.pt SPLIT=train sbatch scripts/precompute_bboxes.sh
#==============================================================================

set -e  # Exit on error

echo "=========================================================================="
echo "YOLO BOUNDING BOX PRECOMPUTATION"
echo "=========================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================================================="

#==============================================================================
# CONFIGURATION
#==============================================================================

# Default paths (can be overridden by environment variables)
YOLO_MODEL="/project/home/p200776/u103235/cvia/CVIA-25-project-team-theta/trained_models/model_weights_yolo/7_yolo11s_640/weights/best.pt"
DATA_ROOT=${DATA_ROOT:-"/project/scratch/p200981/spark2024"}
SPLIT="val"
OUTPUT_DIR="/project/home/p200776/u103235/cvia/CVIA-25-project-team-theta"
BATCH_SIZE=${BATCH_SIZE:-32}
CONFIDENCE=${CONFIDENCE:-0.25}
IOU=${IOU:-0.45}
IMGSZ=${IMGSZ:-640}

echo ""
echo "Configuration:"
echo "  YOLO Model:   $YOLO_MODEL"
echo "  Data root:    $DATA_ROOT"
echo "  Split:        $SPLIT"
echo "  Output dir:   $OUTPUT_DIR"
echo "  Batch size:   $BATCH_SIZE"
echo "  Confidence:   $CONFIDENCE"
echo "  IoU:          $IOU"
echo "  Image size:   $IMGSZ"
echo ""

#==============================================================================
# ENVIRONMENT SETUP
#==============================================================================

# Load modules
module purge
module load Python/3.11.10-GCCcore-13.3.0
module load CUDA/12.6.0
module load OpenMPI/5.0.3-GCC-13.3.0 

source ds_env/bin/activate
pip install ultralytics --quiet 2>/dev/null

# Set CUDA environment
export CUDA_HOME=$EBROOTCUDA
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "Python: $(which python)"
echo "CUDA: $(nvcc --version | grep release)"
echo "GPUs available: $(nvidia-smi -L | wc -l)"
echo ""

#==============================================================================
# RUN PRECOMPUTATION
#==============================================================================

# python src/segformer_optimized/precompute_bboxes.py \
#     --model_path "$YOLO_MODEL" \
#     --data_root "$DATA_ROOT" \
#     --split "$SPLIT" \
#     --output_dir "$OUTPUT_DIR" \
#     --batch_size $BATCH_SIZE \
#     --confidence $CONFIDENCE \
#     --iou $IOU \
#     --imgsz $IMGSZ \
#     --device 0 \
#     --fp16

# Run on GPU node
python3 src/segformer_optimized/precompute_bboxes.py \
    --model_path /project/home/p200776/u103235/cvia/CVIA-25-project-team-theta/trained_models/model_weights_yolo/7_yolo11s_640/weights/best.pt \
    --data_root /project/scratch/p200981/spark2024/ \
    --split train \
    --output_dir /project/home/p200776/u103235/cvia/CVIA-25-project-team-theta

# Verify CSV
python3 scripts/verify_detection_csv.py /project/home/p200776/u103235/cvia/CVIA-25-project-team-theta/train_detection.csv

# Visualize
python3 scripts/visualize_yolo_detections.py

# EXIT_CODE=$?

#==============================================================================
# CLEANUP AND SUMMARY
#==============================================================================

# echo ""
# echo "=========================================================================="
# if [ $EXIT_CODE -eq 0 ]; then
#     echo "PRECOMPUTATION COMPLETED SUCCESSFULLY"
#     echo "=========================================================================="
#     echo "Output CSV: ${OUTPUT_DIR}/${SPLIT}_detection.csv"
#     echo ""
#     echo "You can now use this CSV for training SegFormer Optimized."
#     echo "Update your config file:"
#     echo "  data:"
#     if [ "$SPLIT" = "train" ]; then
#         echo "    train_detection_csv: \"${OUTPUT_DIR}/${SPLIT}_detection.csv\""
#     else
#         echo "    val_detection_csv: \"${OUTPUT_DIR}/${SPLIT}_detection.csv\""
#     fi
# else
#     echo "PRECOMPUTATION FAILED (exit code: $EXIT_CODE)"
# fi
# echo "=========================================================================="
# echo "End time: $(date)"
# echo "=========================================================================="

# exit $EXIT_CODE
