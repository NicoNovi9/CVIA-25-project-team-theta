#!/bin/bash
# Strong Scaling Benchmark for UNet Training
# Maintains fixed global batch size while varying GPU count

# Configuration
RESULTS_DIR="benchmark_results"
RESULTS_FILE="${RESULTS_DIR}/strong_scaling_$(date +%Y%m%d_%H%M%S).csv"
EPOCHS=2  # Short runs for benchmarking

# Create results directory
mkdir -p ${RESULTS_DIR}

# Write CSV header
echo "num_gpus,num_nodes,gpus_per_node,global_batch_size,batch_per_gpu,total_time_sec,avg_epoch_time_sec,samples_per_sec,job_id" > ${RESULTS_FILE}

echo "=========================================="
echo "Strong Scaling Benchmark"
echo "=========================================="
echo "Results will be saved to: ${RESULTS_FILE}"
echo ""

# Strong scaling configurations
# Format: num_gpus:num_nodes:tasks_per_node:batch_per_gpu
# Global batch = 32 fixed (adjust per your memory constraints)
CONFIGS=(
    "1:1:1:32"      # 1 GPU:  32 batch/gpu  = 32 global
    "2:1:2:16"      # 2 GPUs: 16 batch/gpu  = 32 global
    "4:1:4:8"       # 4 GPUs: 8 batch/gpu   = 32 global
    "8:2:4:4"       # 8 GPUs: 4 batch/gpu   = 32 global (2 nodes)
    "16:4:4:2"      # 16 GPUs: 2 batch/gpu  = 32 global (4 nodes)
)

for config in "${CONFIGS[@]}"; do
    IFS=':' read -r num_gpus num_nodes tasks_per_node batch_per_gpu <<< "$config"
    global_batch=$((num_gpus * batch_per_gpu))
    
    echo "=========================================="
    echo "Running: ${num_gpus} GPUs (${num_nodes} nodes, ${tasks_per_node} tasks/node)"
    echo "Batch per GPU: ${batch_per_gpu}, Global batch: ${global_batch}"
    echo "=========================================="
    
    # Submit job
    job_id=$(sbatch --parsable \
        --job-name=bench_${num_gpus}gpu \
        --nodes=${num_nodes} \
        --ntasks-per-node=${tasks_per_node} \
        --gpus-per-task=1 \
        --cpus-per-task=8 \
        --gres=gpu:${tasks_per_node} \
        --time=0:30:00 \
        --partition=gpu \
        --qos=default \
        --account=p200981 \
        --output=${RESULTS_DIR}/bench_${num_gpus}gpu_%j.out \
        --error=${RESULTS_DIR}/bench_${num_gpus}gpu_%j.err \
        --export=ALL,BATCH_SIZE=${batch_per_gpu},N_EPOCHS=${EPOCHS},NUM_GPUS=${num_gpus},NUM_NODES=${num_nodes},GPUS_PER_NODE=${tasks_per_node},GLOBAL_BATCH=${global_batch},RESULTS_FILE=${RESULTS_FILE} \
        --wrap="
source ds_env/bin/activate
module load Python/3.11.10-GCCcore-13.3.0
module load PyTorch/2.3.0-foss-2024a-CUDA-12.6.0
module load torchvision/0.18.1-foss-2024a-CUDA-12.6.0

export CUDA_VISIBLE_DEVICES=0,1,2,3

echo 'Starting benchmark: ${num_gpus} GPUs'
echo 'Batch per GPU: ${batch_per_gpu}'
echo 'Global batch: ${global_batch}'

start_time=\$(date +%s)

srun python -u src/unet/train_unet.py --deepspeed --benchmark

end_time=\$(date +%s)
total_time=\$((end_time - start_time))

# Extract epoch time from output (assuming train_unet.py logs it)
# Parse last line with epoch timing info
avg_epoch_time=\$(grep -oP 'Total Epoch Time: \K[0-9.]+' ${RESULTS_DIR}/bench_${num_gpus}gpu_\${SLURM_JOB_ID}.out | tail -1)
if [ -z \"\$avg_epoch_time\" ]; then
    avg_epoch_time=\$(echo \"scale=2; \$total_time / ${EPOCHS}\" | bc)
fi

# Calculate throughput (samples/sec) - assuming dataset size
# Adjust DATASET_SIZE based on your actual training set
DATASET_SIZE=1000  # Replace with actual size
samples_per_sec=\$(echo \"scale=2; \$DATASET_SIZE * ${EPOCHS} / \$total_time\" | bc)

echo \"${num_gpus},${num_nodes},${tasks_per_node},${global_batch},${batch_per_gpu},\$total_time,\$avg_epoch_time,\$samples_per_sec,\${SLURM_JOB_ID}\" >> ${RESULTS_FILE}

echo 'Benchmark completed in '\$total_time' seconds'
")
    
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
echo "All benchmark jobs submitted!"
echo "=========================================="
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Results will be in: ${RESULTS_FILE}"
echo ""
echo "To view results as they complete:"
echo "  watch -n 5 cat ${RESULTS_FILE}"
