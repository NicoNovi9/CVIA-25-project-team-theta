#!/bin/bash

# # MELUXINA MODULE LOADS--------------------------
# module load env/staging/2025.1
# module load Python/3.13.1-GCCcore-14.2.0
# # -----------------------------------------------

python3 -m venv .venv

source .venv/bin/activate

pip install --upgrade pip

pip install 
pip install torch torchvision numpy pandas

source .venv/bin/activate