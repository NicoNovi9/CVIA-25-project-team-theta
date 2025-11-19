#!/bin/bash -l
#SBATCH --job-name=train 
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --time=02:00:00 
#SBATCH --qos=default 
#SBATCH --account=p200981 
#SBATCH --output=meluxina_train.out
#SBATCH --error=meluxina_train.err

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Starting training script..."
python -u src/train_scripts/train.py