#!/bin/bash -l
#SBATCH --job-name=scaling
#SBATCH --time=20:00:00
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --account=p200776
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=benchmark_results/scaling.out
#SBATCH --error=benchmark_results/scaling.err

# =============================================================================
# Strong Scaling Benchmark for UNet Training
# =============================================================================
# Runs multiple benchmark configurations sequentially to measure strong scaling.
# Maintains fixed global batch size while varying GPU count.
#
# Usage:
#   sbatch scripts/benchmark_strong_scaling.sh
#
# Configuration:
#   DOWNSAMPLE  - Set to 1 to use a small subset of data, 0 for full dataset
#   ZERO_STAGE  - DeepSpeed ZeRO optimization stage (0, 1, 2, or 3)
#   FP16        - Enable FP16 mixed precision (1 = enabled, 0 = disabled)
#   EPOCHS      - Number of training epochs per benchmark run
# =============================================================================

# ----- Configuration -----
DOWNSAMPLE=0   # 1 = use subset for quick testing, 0 = use full dataset
ZERO_STAGE=2   # DeepSpeed ZeRO optimization stage: 0, 1, 2, or 3
FP16=0         # FP16 mixed precision: 1 = enabled, 0 = disabled
EPOCHS=50      # Number of epochs per benchmark run

# ----- Paths -----
RESULTS_DIR="benchmark_results"

# Create results directory
mkdir -p ${RESULTS_DIR}

echo "=========================================="
echo "Strong Scaling Benchmark"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Configuration:"
echo "  DOWNSAMPLE:  ${DOWNSAMPLE}"
echo "  ZERO_STAGE:  ${ZERO_STAGE}"
echo "  FP16:        ${FP16}"
echo "  EPOCHS:      ${EPOCHS}"
echo ""
echo "Results will be saved to: ${RESULTS_DIR}/training_summary.csv"
echo "=========================================="
echo ""

# Strong scaling configurations
# Format: num_gpus:num_nodes:tasks_per_node:batch_per_gpu
# STRONG SCALING: Fixed global batch = 32
CONFIGS=(
    # "1:1:1:32"      # 1 GPU:   32 batch/gpu = 32 global
    # "2:1:2:16"      # 2 GPUs:  16 batch/gpu = 32 global
    # "4:1:4:8"       # 4 GPUs:  8 batch/gpu  = 32 global
    # "8:2:4:4"       # 8 GPUs:  4 batch/gpu  = 32 global (2 nodes)
    # "1:1:1:8"       # 1 GPU:   8 batch/gpu  = 8 global
    # "2:1:2:8"       # 2 GPUs:  8 batch/gpu  = 16 global
    # "4:1:4:8"       # 4 GPUs:  8 batch/gpu  = 32 global
    # "8:2:4:8"       # 8 GPUs:  8 batch/gpu  = 64 global (2 nodes)
    # "16:4:4:8"      # 16 GPUs: 8 batch/gpu  = 128 global (4 nodes)
    "16:4:4:16"      # 16 GPUs: 16 batch/gpu = 256 global (4 nodes)
)

for config in "${CONFIGS[@]}"; do
    IFS=':' read -r num_gpus num_nodes tasks_per_node batch_per_gpu <<< "$config"
    global_batch=$((num_gpus * batch_per_gpu))
    
    echo "=========================================="
    echo "Running: ${num_gpus} GPUs (${num_nodes} nodes, ${tasks_per_node} tasks/node)"
    echo "Batch per GPU: ${batch_per_gpu}, Global batch: ${global_batch}"
    echo "=========================================="
    
    # Submit job using the same script as benchmark_single.sh
    job_id=$(sbatch --parsable \
        --job-name=${num_gpus}gpu \
        --nodes=${num_nodes} \
        --ntasks-per-node=${tasks_per_node} \
        --gpus-per-task=1 \
        --cpus-per-task=8 \
        --gres=gpu:${tasks_per_node} \
        --time=0:30:00 \
        --partition=gpu \
        --qos=default \
        --account=p200776 \
        --output=${RESULTS_DIR}/${num_gpus}gpu_%j.out \
        --error=${RESULTS_DIR}/${num_gpus}gpu_%j.err \
        --export=ALL,BATCH_SIZE=${batch_per_gpu},N_EPOCHS=${EPOCHS},DOWNSAMPLE=${DOWNSAMPLE},ZERO_STAGE=${ZERO_STAGE},FP16=${FP16} \
        scripts/unet_train.sh)
    
    echo "Job submitted: ${job_id}"
    echo ""
    
    # Wait for job to complete before submitting next one
    echo "Waiting for job ${job_id} to complete..."
    while squeue -j ${job_id} 2>/dev/null | grep -q ${job_id}; do
        sleep 10
    done
    echo "Job ${job_id} completed."
    echo ""
done

echo "=========================================="
echo "All benchmark jobs completed!"
echo "=========================================="
echo "End time: $(date)"
echo ""
echo "Results are in: ${RESULTS_DIR}/training_summary.csv"
echo ""
echo "To view results:"
echo "  cat ${RESULTS_DIR}/training_summary.csv"
