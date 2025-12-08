#!/bin/bash -l
#SBATCH --job-name=inference_detr
#SBATCH --output=detr_inference.out
#SBATCH --error=detr_inference.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --time=01:00:00
#SBATCH --account=p200981

echo "=================================================="
echo "DETR Detection Inference"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=================================================="

# Environment setup
module load Python/3.11.10-GCCcore-13.3.0
module load scikit-learn/1.5.2-gfbf-2024a
module load matplotlib/3.9.2-gfbf-2024a
module load Seaborn/0.13.2-gfbf-2024a

module load PyTorch/2.3.0-foss-2024a-CUDA-12.6.0
module load torchvision/0.18.1-foss-2024a-CUDA-12.6.0

if [ ! -d "ds_env" ]; then
    python -m venv ds_env

    source ds_env/bin/activate

    pip install --upgrade pip
    pip install setuptools deepspeed mpi4py
    pip install transformers accelerate timm
    pip install pandas pycocotools
fi

source ds_env/bin/activate

# Make all 4 GPUs visible
export CUDA_VISIBLE_DEVICES=0,1,2,3

cd ${SLURM_SUBMIT_DIR}

# Paths configuration
MODEL="model_weights/detr_best"
DATA="/project/scratch/p200981/spark2024_test/detection/images"
OUTPUT_DIR="submission_output"

echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Data: $DATA"
echo "  Output directory: $OUTPUT_DIR"
echo "  GPUs: 4 (DataParallel)"
echo ""

# Run DETR detection inference
python -u src/detr/inference_detr.py \
    --model_path "$MODEL" \
    --data_path "$DATA" \
    --output_dir "$OUTPUT_DIR" \
    --threshold 0.3 \
    --batch_size 32 \
    --num_workers 7

echo ""
echo "=================================================="
echo "Job completed: $(date)"
echo "=================================================="
echo ""
echo "Output files in: $OUTPUT_DIR"
echo "Submission zip: $OUTPUT_DIR/detection_submission.zip"
