#!/bin/bash
# --------------------------------------------------------
# LSF Multi-GPU Training Script (Simplified Version)
# Using PyTorch's torch.distributed.launch
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

# Description:
# Simplified script for single-node multi-GPU training
# Default: 4 GPUs on 1 node
# For multi-node training, use lsf_train_multi_gpu.sh

# ========================================
# Configuration
# ========================================

# Training script
TRAIN_SCRIPT="train_semantic_softmax.py"  # or "train_single_label.py"

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
NGPUS=$(nvidia-smi --list-gpus | wc -l)
if [ -z "$NGPUS" ]; then
    NGPUS=4  # Default fallback
fi

# Output
OUTPUT_DIR="./output/${MODEL_NAME}_$(date +%Y%m%d_%H%M%S)"

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

# ========================================
# Info
# ========================================

echo "=========================================="
echo "ImageNet-21K Single-Node Multi-GPU Training"
echo "=========================================="
echo "Job ID: $LSB_JOBID"
echo "Node: $(hostname)"
echo "GPUs: $NGPUS"
echo "Model: $MODEL_NAME"
echo "Batch Size per GPU: $BATCH_SIZE"
echo "Effective Batch Size: $((BATCH_SIZE * NGPUS))"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# ========================================
# Launch Training
# ========================================

if [ "$TRAIN_SCRIPT" = "train_semantic_softmax.py" ]; then
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
        2>&1 | tee ${OUTPUT_DIR}/training.log
fi

echo "Training completed at: $(date)"
echo "Output saved to: $OUTPUT_DIR"
