#!/bin/bash -l
#SBATCH --job-name=eval_yolo
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --account=p200981

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16

#SBATCH --output=yolo_eval.out
#SBATCH --error=yolo_eval.err

echo "=================================================="
echo "YOLO Evaluation - SPARK Detection Dataset"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=================================================="

# Load modules
module load Python/3.11.10-GCCcore-13.3.0
# module load scikit-learn/1.5.2-gfbf-2024a
# module load matplotlib/3.9.2-gfbf-2024a

# module load PyTorch/2.3.0-foss-2024a-CUDA-12.6.0
# module load torchvision/0.18.1-foss-2024a-CUDA-12.6.0

module load CUDA/12.6.0
module load OpenMPI/5.0.3-GCC-13.3.0 

source ds_env/bin/activate

# Install dependencies if needed
# pip install ultralytics --quiet 2>/dev/null

export CUDA_VISIBLE_DEVICES=0,1,2,3

cd ${SLURM_SUBMIT_DIR}

# === CONFIGURATION ===
# Change these paths as needed
MODEL_PATH="/project/home/p200776/u103235/cvia/CVIA-25-project-team-theta/trained_models/model_weights_yolo/7_yolo11s_640/weights/best.pt"
DATA_YAML="/project/scratch/p200981/spark2024_yolo/data.yaml"
OUTPUT_DIR="evaluation_results_yolo_temp"

echo ""
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Data: $DATA_YAML"
echo "  Output: $OUTPUT_DIR"
echo ""

# Run evaluation
python -u src/yolo/eval_yolo.py \
    --model_path "$MODEL_PATH" \
    --data_yaml "$DATA_YAML" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 32 \
    --imgsz 640 \
    --device 0

echo ""
echo "=================================================="
echo "Job completed: $(date)"
echo "=================================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
