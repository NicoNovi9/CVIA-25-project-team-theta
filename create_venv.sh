#!/bin/bash

# MELUXINA MODULE LOADS--------------------------
# this is necessary to have the right python version
# the already available in meluxina is too old and 
# if we don't load a newer one, also the pytorch version
# that pip installs is too old and doesn't support cuda with
# sm_80 architecture (needed for A100 GPUs)
module load Python
# -----------------------------------------------

python -m venv .venv

source .venv/bin/activate

pip install --upgrade pip
pip install torch torchvision numpy pandas

source .venv/bin/activate