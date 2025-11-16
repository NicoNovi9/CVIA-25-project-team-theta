#!/bin/bash 
#SBATCH --job-name=monitoring_stack #SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=2 
#SBATCH --mem=8G 
#SBATCH --time=02:00:00 
#SBATCH --qos=default 
#SBATCH --partition=cpu 
#SBATCH --account=p200981 
#SBATCH --output=output/logs/monitoring_stack.out 
#SBATCH --error=output/logs/monitoring_stack.err