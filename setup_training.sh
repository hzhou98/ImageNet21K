#!/bin/bash
# --------------------------------------------------------
# Setup script for ImageNet-21K training on LSF cluster
# Downloads pretrained models and prepares environment
# --------------------------------------------------------

set -e  # Exit on error

echo "=========================================="
echo "ImageNet-21K Training Setup"
echo "=========================================="

# ========================================
# Configuration
# ========================================

RESOURCE_DIR="./resources"
PRETRAINED_DIR="./pretrained_models"
LOG_DIR="./logs"
OUTPUT_DIR="./output"

# ========================================
# Create Directories
# ========================================

echo "Creating directory structure..."
mkdir -p ${RESOURCE_DIR}
mkdir -p ${PRETRAINED_DIR}
mkdir -p ${LOG_DIR}
mkdir -p ${OUTPUT_DIR}

# ========================================
# Download Semantic Tree
# ========================================

echo ""
echo "Downloading semantic tree for fall11 dataset..."
if [ ! -f "${RESOURCE_DIR}/imagenet21k_miil_tree.pth" ]; then
    wget -O ${RESOURCE_DIR}/imagenet21k_miil_tree.pth \
        https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/resources/fall11/imagenet21k_miil_tree.pth
    echo "✓ Semantic tree downloaded"
else
    echo "✓ Semantic tree already exists"
fi

# Optional: Download winter21 tree
read -p "Download winter21 semantic tree? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ ! -f "${RESOURCE_DIR}/imagenet21k_miil_tree_winter21.pth" ]; then
        wget -O ${RESOURCE_DIR}/imagenet21k_miil_tree_winter21.pth \
            https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/resources/winter21/imagenet21k_miil_tree.pth
        echo "✓ Winter21 semantic tree downloaded"
    fi
fi

# ========================================
# Download Pretrained Models (ImageNet-1K)
# ========================================

echo ""
echo "Setting up pretrained models..."
echo "These models are used as initialization for faster training."
echo ""

# Function to download from timm
download_timm_model() {
    local model_name=$1
    local output_file="${PRETRAINED_DIR}/${model_name}_1k.pth"
    
    if [ ! -f "$output_file" ]; then
        echo "Downloading ${model_name}..."
        python3 << EOF
import torch
import timm

try:
    model = timm.create_model('$model_name', pretrained=True)
    torch.save(model.state_dict(), '$output_file')
    print(f"✓ $model_name downloaded successfully")
except Exception as e:
    print(f"✗ Failed to download $model_name: {e}")
    exit(1)
EOF
    else
        echo "✓ ${model_name} already exists"
    fi
}

# Download popular models
echo "Available models to download:"
echo "1. tresnet_m (recommended, 31M params)"
echo "2. tresnet_l (55M params, higher accuracy)"
echo "3. resnet50 (25M params, baseline)"
echo "4. mobilenetv3_large_100 (5.5M params, fast)"
echo "5. vit_base_patch16_224 (86M params, transformer)"
echo "6. All of the above"
echo "7. Skip (download manually later)"
echo ""
read -p "Select option (1-7): " choice

case $choice in
    1)
        download_timm_model "tresnet_m"
        ;;
    2)
        download_timm_model "tresnet_l"
        ;;
    3)
        download_timm_model "resnet50"
        ;;
    4)
        download_timm_model "mobilenetv3_large_100"
        ;;
    5)
        download_timm_model "vit_base_patch16_224"
        ;;
    6)
        download_timm_model "tresnet_m"
        download_timm_model "tresnet_l"
        download_timm_model "resnet50"
        download_timm_model "mobilenetv3_large_100"
        download_timm_model "vit_base_patch16_224"
        ;;
    7)
        echo "Skipping model download"
        ;;
    *)
        echo "Invalid choice, skipping"
        ;;
esac

# ========================================
# Check Python Environment
# ========================================

echo ""
echo "Checking Python environment..."

# Check if required packages are installed
python3 << EOF
import sys

required_packages = [
    'torch',
    'torchvision',
    'timm',
    'PIL',
    'numpy'
]

missing_packages = []

for package in required_packages:
    try:
        __import__(package)
        print(f"✓ {package} is installed")
    except ImportError:
        print(f"✗ {package} is NOT installed")
        missing_packages.append(package)

if missing_packages:
    print(f"\nMissing packages: {', '.join(missing_packages)}")
    print("Install with: pip install torch torchvision timm Pillow numpy")
    print("Or: pip install -r requirements.txt")
    sys.exit(1)
else:
    print("\n✓ All required packages are installed")
EOF

# ========================================
# Check Data Path
# ========================================

echo ""
echo "Checking data path..."
DATA_PATH="/research/groups/yu3grp/projects/scRNASeq/yu3grp/hzhou98/PublicDataSets/ImageNet21K/fall11"

if [ -d "$DATA_PATH" ]; then
    # Count directories (classes)
    NUM_CLASSES=$(find $DATA_PATH -mindepth 1 -maxdepth 1 -type d | wc -l)
    echo "✓ Data path exists: $DATA_PATH"
    echo "  Number of classes: $NUM_CLASSES"
    
    if [ $NUM_CLASSES -lt 10000 ]; then
        echo "  ⚠ Warning: Expected ~11221 classes, found $NUM_CLASSES"
        echo "  Make sure dataset is properly processed"
    fi
else
    echo "✗ Data path does NOT exist: $DATA_PATH"
    echo "  Please process the dataset first using:"
    echo "  cd dataset_preprocessing && bash processing_script_new.sh"
fi

# ========================================
# Check GPU Availability
# ========================================

echo ""
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    NGPUS=$(nvidia-smi --list-gpus | wc -l)
    echo "✓ nvidia-smi found"
    echo "  Available GPUs: $NGPUS"
    
    if [ $NGPUS -eq 0 ]; then
        echo "  ⚠ No GPUs detected on this node"
        echo "  (This is normal on login nodes)"
    else
        echo ""
        nvidia-smi --query-gpu=index,name,memory.total --format=csv
    fi
else
    echo "✗ nvidia-smi not found"
    echo "  (This is normal on login nodes)"
fi

# ========================================
# Make Scripts Executable
# ========================================

echo ""
echo "Making LSF scripts executable..."
chmod +x lsf_train_*.sh 2>/dev/null || true
echo "✓ Scripts are executable"

# ========================================
# Summary
# ========================================

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Directory structure:"
echo "  ${RESOURCE_DIR}/    - Semantic trees"
echo "  ${PRETRAINED_DIR}/  - ImageNet-1K pretrained models"
echo "  ${LOG_DIR}/         - Job output logs"
echo "  ${OUTPUT_DIR}/      - Training outputs and checkpoints"
echo ""
echo "Next steps:"
echo "1. Review and edit LSF training scripts:"
echo "   - lsf_train_single_node.sh   (single node, 4 GPUs)"
echo "   - lsf_train_multi_gpu.sh     (multi-node)"
echo "   - lsf_train_multinode.sh     (with jsrun/mpirun)"
echo ""
echo "2. Adjust these parameters in the script:"
echo "   - MODEL_NAME: tresnet_m, resnet50, etc."
echo "   - BATCH_SIZE: based on your GPU memory"
echo "   - DATA_PATH: verify it points to your dataset"
echo ""
echo "3. Submit job:"
echo "   bsub < lsf_train_single_node.sh"
echo ""
echo "4. Monitor job:"
echo "   bjobs"
echo "   tail -f logs/train_<JOBID>.out"
echo ""
echo "For detailed instructions, see LSF_TRAINING_GUIDE.md"
echo ""
