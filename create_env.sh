#!/bin/bash

python3 -m venv spark_venv

source spark_venv/bin/activate

pip install --upgrade pip

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo "Virtualenv 'spark_venv' creata e PyTorch installato."
