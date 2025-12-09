#!/bin/bash
# =============================================================================
# Benchmark Runner Script
# =============================================================================
# This script launches a single benchmark run for UNet training with DeepSpeed.
# It submits a SLURM job with the specified GPU configuration.
#
# Usage:
#   ./benchmark_single.sh <num_gpus> <num_nodes> <tasks_per_node> <batch_per_gpu> <epochs>
#
# Arguments:
#   num_gpus       - Total number of GPUs (default: 1)
#   num_nodes      - Number of nodes to use (default: 1)
#   tasks_per_node - Number of tasks (GPUs) per node (default: 1)
#   batch_per_gpu  - Batch size per GPU (default: 32)
#   epochs         - Number of training epochs (default: 1)
#
# Examples:
#   ./benchmark_single.sh 1 1 1 32 1    # 1 GPU, batch 32, 1 epoch
#   ./benchmark_single.sh 4 1 4 16 5    # 4 GPUs on 1 node, batch 16, 5 epochs
#   ./benchmark_single.sh 8 2 4 8 10    # 8 GPUs on 2 nodes (4 per node), batch 8, 10 epochs
#
# Configuration:
#   DOWNSAMPLE - Set to 1 to use a small subset of data (200 samples) for quick tests
#                Set to 0 to use the full dataset
#   ZERO_STAGE - DeepSpeed ZeRO optimization stage (0, 1, 2, or 3)
#   FP16       - Enable FP16 mixed precision (1 = enabled, 0 = disabled)
# =============================================================================

# ----- Configuration -----
DOWNSAMPLE=0  # 1 = use 200 samples for quick testing, 0 = use full dataset
ZERO_STAGE=2  # DeepSpeed ZeRO optimization stage: 0, 1, 2, or 3
FP16=0        # FP16 mixed precision: 1 = enabled, 0 = disabled

# ----- Arguments -----
NUM_GPUS=${1:-4}
NUM_NODES=${2:-1}
TASKS_PER_NODE=${3:-4}
BATCH_PER_GPU=${4:-8}
EPOCHS=${5:-2}

GLOBAL_BATCH=$((NUM_GPUS * BATCH_PER_GPU))
RESULTS_DIR="benchmark_results"
mkdir -p ${RESULTS_DIR}

echo "=========================================="
echo "Single Benchmark Run"
echo "=========================================="
echo "GPUs: ${NUM_GPUS} (${NUM_NODES} nodes x ${TASKS_PER_NODE} tasks/node)"
echo "Batch per GPU: ${BATCH_PER_GPU}"
echo "Global batch: ${GLOBAL_BATCH}"
echo "Epochs: ${EPOCHS}"
echo "=========================================="

sbatch --job-name=bench_${NUM_GPUS}gpu \
    --nodes=${NUM_NODES} \
    --ntasks-per-node=${TASKS_PER_NODE} \
    --gpus-per-task=1 \
    --cpus-per-task=8 \
    --gres=gpu:${TASKS_PER_NODE} \
    --time=0:30:00 \
    --partition=gpu \
    --qos=default \
    --account=p200776 \
    --output=${RESULTS_DIR}/bench_${NUM_GPUS}gpu_%j.out \
    --error=${RESULTS_DIR}/bench_${NUM_GPUS}gpu_%j.err \
    --export=ALL,BATCH_SIZE=${BATCH_PER_GPU},N_EPOCHS=${EPOCHS},DOWNSAMPLE=${DOWNSAMPLE},ZERO_STAGE=${ZERO_STAGE},FP16=${FP16} \
    scripts/unet_train.sh

    # p200776
    # p200981
    # --mail-type=BEGIN,END,FAIL \
    # --mail-user=nicolanoventa9@gmail.com \
echo ""
echo "Job submitted. Check output in ${RESULTS_DIR}/"
