#!/bin/bash
# Quick benchmark runner - manually launch one configuration at a time
# Usage: ./benchmark_single.sh <num_gpus> <num_nodes> <tasks_per_node> <batch_per_gpu>
# Example: ./benchmark_single.sh 4 1 4 8

NUM_GPUS=${1:-1}
NUM_NODES=${2:-1}
TASKS_PER_NODE=${3:-1}
BATCH_PER_GPU=${4:-32}
EPOCHS=${5:-1}

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
    --account=p200981 \
    --output=${RESULTS_DIR}/bench_${NUM_GPUS}gpu_%j.out \
    --error=${RESULTS_DIR}/bench_${NUM_GPUS}gpu_%j.err \
    --export=ALL,BATCH_SIZE=${BATCH_PER_GPU},N_EPOCHS=${EPOCHS} \
    scripts/unet_train.sh


    # --mail-type=BEGIN,END,FAIL \
    # --mail-user=nicolanoventa9@gmail.com \
echo ""
echo "Job submitted. Check output in ${RESULTS_DIR}/"
