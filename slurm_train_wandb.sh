#!/bin/bash
# --------------------------------------------------------
# Slurm Single-Node Training with W&B Support
# --------------------------------------------------------

#SBATCH --job-name=imagenet21k_wandb
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --partition=gpu

# ========================================
# Configuration
# ========================================

# Training script (W&B-enabled)
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
NGPUS=${SLURM_GPUS_ON_NODE:-4}
OUTPUT_DIR="./output/${MODEL_NAME}_$(date +%Y%m%d_%H%M%S)"

# ========================================
# Weights & Biases Configuration
# ========================================

USE_WANDB=true
WANDB_PROJECT="imagenet21k-training"
WANDB_ENTITY=""  # Your W&B username or team
WANDB_RUN_NAME="${MODEL_NAME}_${NGPUS}gpus_$(date +%Y%m%d_%H%M%S)"

# W&B API Key (set via environment or login once)
# export WANDB_API_KEY="your_api_key_here"

# ========================================
# Setup
# ========================================

mkdir -p ${OUTPUT_DIR}
mkdir -p logs

module load cuda/11.7
module load gcc/9.3.0
module load python/3.9

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NCCL_DEBUG=INFO
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# W&B settings
if [ "$USE_WANDB" = true ]; then
    export WANDB_MODE=online
    export WANDB_CACHE_DIR="${OUTPUT_DIR}/.wandb_cache"
    export WANDB_DIR="${OUTPUT_DIR}"
fi

# ========================================
# Info
# ========================================

echo "=========================================="
echo "ImageNet-21K Training with W&B (Slurm)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
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

WANDB_ARGS=""
if [ "$USE_WANDB" = true ]; then
    WANDB_ARGS="--use_wandb --wandb_project=$WANDB_PROJECT --wandb_run_name=$WANDB_RUN_NAME"
    if [ ! -z "$WANDB_ENTITY" ]; then
        WANDB_ARGS="$WANDB_ARGS --wandb_entity=$WANDB_ENTITY"
    fi
fi

if [[ "$TRAIN_SCRIPT" == *"semantic"* ]]; then
    python -m torch.distributed.launch \
        --nproc_per_node=$NGPUS \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
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
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
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
if [ "$USE_WANDB" = true ]; then
    echo "View results at: https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT"
fi
