#!/bin/bash
# --------------------------------------------------------
# LSF Single-Node Multi-GPU Training with W&B Support
# --------------------------------------------------------

#BSUB -P ImageNet21K_Training
#BSUB -J imagenet21k_train
#BSUB -o logs/train_%J.out
#BSUB -e logs/train_%J.err
#BSUB -W 48:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=4:mode=exclusive_process:mps=no:j_exclusive=yes"
#BSUB -q gpu
#BSUB -R "rusage[mem=32000]"

# ========================================
# Configuration
# ========================================

# Training script (W&B-enabled versions)
TRAIN_SCRIPT="train_semantic_softmax_wandb.py"  # or "train_single_label_wandb.py"

# Paths
DATA_PATH="/research/groups/yu3grp/projects/scRNASeq/yu3grp/hzhou98/PublicDataSets/ImageNet21K/fall11"
TREE_PATH="./resources/imagenet21k_miil_tree.pth"

# Model settings
MODEL_NAME="tresnet_m"
MODEL_PATH="./pretrained_models/${MODEL_NAME}_1k.pth"

# Hyperparameters
BATCH_SIZE=64
EPOCHS=80
LR=3e-4
WEIGHT_DECAY=1e-4
LABEL_SMOOTH=0.2
NUM_WORKERS=8
IMAGE_SIZE=224
NUM_CLASSES=11221

# GPU configuration
NGPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
if [ -z "$NGPUS" ] || [ "$NGPUS" -eq 0 ]; then
    NGPUS=4  # Default fallback
fi

# Output
OUTPUT_DIR="./output/${MODEL_NAME}_$(date +%Y%m%d_%H%M%S)"

# ========================================
# Weights & Biases Configuration
# ========================================

# Enable W&B (set to false to disable)
USE_WANDB=true

# W&B settings
WANDB_PROJECT="imagenet21k-training"
WANDB_ENTITY=""  # Your W&B username or team name (leave empty for default)
WANDB_RUN_NAME="${MODEL_NAME}_${NGPUS}gpus_$(date +%Y%m%d_%H%M%S)"

# W&B API Key (required for first run)
# Set your API key here or via environment variable
# Get your key from: https://wandb.ai/authorize
# export WANDB_API_KEY="your_api_key_here"

# Alternative: Login once on the cluster with: wandb login

# ========================================
# Setup
# ========================================

mkdir -p ${OUTPUT_DIR}
mkdir -p logs

# Load modules (adjust for your cluster)
module load cuda/11.7
module load gcc/9.3.0
module load python/3.9

# Environment variables
export OMP_NUM_THREADS=4
export NCCL_DEBUG=INFO

# W&B environment variables
if [ "$USE_WANDB" = true ]; then
    # Set W&B mode (online, offline, or disabled)
    export WANDB_MODE=online  # Change to 'offline' for offline logging
    
    # W&B cache directory (optional - useful for cluster environments)
    export WANDB_CACHE_DIR="${OUTPUT_DIR}/.wandb_cache"
    export WANDB_DIR="${OUTPUT_DIR}"
    
    # Disable W&B system stats if needed (reduces overhead)
    # export WANDB_DISABLE_SYSTEM_STATS=true
fi

# ========================================
# Info
# ========================================

echo "=========================================="
echo "ImageNet-21K Training with W&B"
echo "=========================================="
echo "Job ID: $LSB_JOBID"
echo "Node: $(hostname)"
echo "GPUs: $NGPUS"
echo "Model: $MODEL_NAME"
echo "Batch Size per GPU: $BATCH_SIZE"
echo "Effective Batch Size: $((BATCH_SIZE * NGPUS))"
echo "Weights & Biases: $USE_WANDB"
if [ "$USE_WANDB" = true ]; then
    echo "W&B Project: $WANDB_PROJECT"
    echo "W&B Run Name: $WANDB_RUN_NAME"
fi
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# ========================================
# Launch Training
# ========================================

# Build W&B arguments
WANDB_ARGS=""
if [ "$USE_WANDB" = true ]; then
    WANDB_ARGS="--use_wandb"
    WANDB_ARGS="$WANDB_ARGS --wandb_project=$WANDB_PROJECT"
    WANDB_ARGS="$WANDB_ARGS --wandb_run_name=$WANDB_RUN_NAME"
    if [ ! -z "$WANDB_ENTITY" ]; then
        WANDB_ARGS="$WANDB_ARGS --wandb_entity=$WANDB_ENTITY"
    fi
fi

if [[ "$TRAIN_SCRIPT" == *"semantic"* ]]; then
    python -m torch.distributed.launch \
        --nproc_per_node=$NGPUS \
        --master_port=29500 \
        $TRAIN_SCRIPT \
        --data_path=$DATA_PATH \
        --tree_path=$TREE_PATH \
        --model_name=$MODEL_NAME \
        --model_path=$MODEL_PATH \
        --batch_size=$BATCH_SIZE \
        --epochs=$EPOCHS \
        --lr=$LR \
        --weight_decay=$WEIGHT_DECAY \
        --label_smooth=$LABEL_SMOOTH \
        --num_workers=$NUM_WORKERS \
        --image_size=$IMAGE_SIZE \
        --num_classes=$NUM_CLASSES \
        $WANDB_ARGS \
        2>&1 | tee ${OUTPUT_DIR}/training.log
else
    python -m torch.distributed.launch \
        --nproc_per_node=$NGPUS \
        --master_port=29500 \
        $TRAIN_SCRIPT \
        --data_path=$DATA_PATH \
        --model_name=$MODEL_NAME \
        --model_path=$MODEL_PATH \
        --batch_size=$BATCH_SIZE \
        --epochs=$EPOCHS \
        --lr=$LR \
        --weight_decay=$WEIGHT_DECAY \
        --label_smooth=$LABEL_SMOOTH \
        --num_workers=$NUM_WORKERS \
        --image_size=$IMAGE_SIZE \
        --num_classes=$NUM_CLASSES \
        $WANDB_ARGS \
        2>&1 | tee ${OUTPUT_DIR}/training.log
fi

echo "Training completed at: $(date)"
echo "Output saved to: $OUTPUT_DIR"

if [ "$USE_WANDB" = true ]; then
    echo ""
    echo "View your training metrics at:"
    echo "https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT"
fi
