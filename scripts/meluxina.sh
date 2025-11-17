#!/bin/bash -l
#SBATCH --job-name=train 
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --partition=gpu 
#SBATCH --time=02:00:00 
#SBATCH --qos=default 
#SBATCH --account=p200981 

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Starting training script..."
# python src/test_pytorch_version.py
python -u src/train.py