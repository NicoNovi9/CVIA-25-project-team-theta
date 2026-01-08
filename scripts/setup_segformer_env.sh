#!/bin/bash
set -e  # fail fast

echo "=== SegFormer environment setup (HPC-safe) ==="

# ------------------------------------------------------------------
# 1. Clean module environment
# ------------------------------------------------------------------
module purge

# ------------------------------------------------------------------
# 2. Load required base modules ONLY
# ------------------------------------------------------------------
module load Python/3.11.10-GCCcore-13.3.0
module load CUDA/12.6.0
module load OpenMPI/5.0.3-GCCcore-13.3.0

echo "Loaded modules:"
module list

# ------------------------------------------------------------------
# 3. Create fresh virtual environment
# ------------------------------------------------------------------
ENV_NAME="ds_env"

if [ -d "$ENV_NAME" ]; then
    echo "Removing existing virtual environment: $ENV_NAME"
    rm -rf "$ENV_NAME"
fi

echo "Creating virtual environment..."
python -m venv "$ENV_NAME"
source "$ENV_NAME/bin/activate"

# ------------------------------------------------------------------
# 4. Upgrade core Python tooling
# ------------------------------------------------------------------
echo "Upgrading pip / setuptools / wheel..."
pip install --upgrade pip setuptools wheel

# ------------------------------------------------------------------
# 5. Install PyTorch >= 2.6 (CUDA 12.6)
# ------------------------------------------------------------------
echo "Installing PyTorch (CUDA 12.6)..."
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu126

# ------------------------------------------------------------------
# 6. Install remaining dependencies
# ------------------------------------------------------------------
echo "Installing Python dependencies..."
pip install \
  transformers \
  accelerate \
  timm \
  deepspeed \
  mpi4py \
  pandas \
  pycocotools \
  pyyaml \
  scikit-learn \
  matplotlib \
  seaborn

# ------------------------------------------------------------------
# 7. Verification: Python / Torch / CUDA / Transformers
# ------------------------------------------------------------------
echo "=== Verifying core stack ==="
python - << 'EOF'
import torch, transformers, torchvision

print("Python        :", torch.sys.version.split()[0])
print("Torch         :", torch.__version__)
print("Torch CUDA    :", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count     :", torch.cuda.device_count())
print("Transformers  :", transformers.__version__)
print("Torchvision   :", torchvision.__version__)
EOF

# ------------------------------------------------------------------
# 8. Verification: MPI runtime
# ------------------------------------------------------------------
echo "=== Verifying MPI ==="
python - << 'EOF'
from mpi4py import MPI
print("MPI vendor:", MPI.get_vendor())
print("MPI size  :", MPI.COMM_WORLD.Get_size())
EOF

# ------------------------------------------------------------------
# 9. Verification: SegFormer model loading
# ------------------------------------------------------------------
echo "=== Verifying SegFormer loading ==="
python - << 'EOF'
from transformers import SegformerForSemanticSegmentation
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512"
)
print("SegFormer loaded successfully")
EOF

echo "=== Environment setup completed successfully ==="
echo "Activate it with: source ds_env/bin/activate"
