#!/bin/bash
# --------------------------------------------------------
# LSF Multi-Node Multi-GPU Training Script
# Using modern PyTorch distributed training
# --------------------------------------------------------

#BSUB -P ImageNet21K_Training
#BSUB -J imagenet21k_multinode
#BSUB -o logs/train_%J.out
#BSUB -e logs/train_%J.err
#BSUB -W 72:00
#BSUB -nnodes 4
#BSUB -alloc_flags gpumps
#BSUB -q gpu
#BSUB -R "rusage[mem=32000]"

# Description:
# Multi-node distributed training using LSF jsrun
# Default: 4 nodes, all available GPUs per node
# Adjust -nnodes based on your needs

# ========================================
# Configuration
# ========================================

TRAIN_SCRIPT="train_semantic_softmax.py"
DATA_PATH="/research/groups/yu3grp/projects/scRNASeq/yu3grp/hzhou98/PublicDataSets/ImageNet21K/fall11"
TREE_PATH="./resources/imagenet21k_miil_tree.pth"

MODEL_NAME="tresnet_m"
MODEL_PATH="./pretrained_models/${MODEL_NAME}_1k.pth"

BATCH_SIZE=64
EPOCHS=80
LR=3e-4
WEIGHT_DECAY=1e-4
LABEL_SMOOTH=0.2
NUM_WORKERS=8
IMAGE_SIZE=224
NUM_CLASSES=11221

OUTPUT_DIR="./output/${MODEL_NAME}_$(date +%Y%m%d_%H%M%S)"

# ========================================
# Setup
# ========================================

mkdir -p ${OUTPUT_DIR}
mkdir -p logs

module load cuda/11.7
module load gcc/9.3.0
module load python/3.9

export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=4

# Get node information
NNODES=$(echo $LSB_MCPU_HOSTS | wc -w)
NNODES=$((NNODES / 2))  # LSF lists host and core count

echo "=========================================="
echo "Multi-Node Training Configuration"
echo "=========================================="
echo "Nodes: $NNODES"
echo "Model: $MODEL_NAME"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# ========================================
# Launch with jsrun (for IBM LSF systems)
# ========================================

# If your cluster supports jsrun
if command -v jsrun &> /dev/null; then
    if [ "$TRAIN_SCRIPT" = "train_semantic_softmax.py" ]; then
        jsrun -n $NNODES -r 1 -a 1 -c 42 -g ALL_GPUS \
            python -m torch.distributed.launch \
            --nproc_per_node=ALL_GPUS \
            --nnodes=$NNODES \
            --node_rank=\$PMIX_RANK \
            --master_addr=$(hostname) \
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
        jsrun -n $NNODES -r 1 -a 1 -c 42 -g ALL_GPUS \
            python -m torch.distributed.launch \
            --nproc_per_node=ALL_GPUS \
            --nnodes=$NNODES \
            --node_rank=\$PMIX_RANK \
            --master_addr=$(hostname) \
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
else
    echo "jsrun not available. Using alternative launch method..."
    # Alternative for clusters without jsrun - using mpirun
    if [ "$TRAIN_SCRIPT" = "train_semantic_softmax.py" ]; then
        mpirun -np $NNODES \
            python -m torch.distributed.launch \
            --nproc_per_node=4 \
            --nnodes=$NNODES \
            --node_rank=\$OMPI_COMM_WORLD_RANK \
            --master_addr=$(hostname) \
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
    fi
fi

echo "Training completed: $(date)"
