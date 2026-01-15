#!/bin/bash
# --------------------------------------------------------
# LSF Multi-GPU Training Script for ImageNet-21K
# Distributed training using PyTorch DistributedDataParallel
# --------------------------------------------------------

#BSUB -P ImageNet21K_Training
#BSUB -J imagenet21k_train
#BSUB -o logs/train_%J.out
#BSUB -e logs/train_%J.err
#BSUB -W 48:00
#BSUB -n 32
#BSUB -R "span[ptile=8]"
#BSUB -gpu "num=4:mode=exclusive_process:mps=no:j_exclusive=yes"
#BSUB -q gpu
#BSUB -R "rusage[mem=16000]"

# Description:
# This script trains ImageNet-21K models using multiple GPUs across multiple nodes
# Default configuration: 4 nodes x 4 GPUs = 16 GPUs total
# Adjust -n, -R span[ptile=X], and -gpu num=X based on your cluster resources

# ========================================
# Configuration Parameters
# ========================================

# Training script to use (choose one)
TRAIN_SCRIPT="train_semantic_softmax.py"  # or "train_single_label.py"

# Data paths
DATA_PATH="/research/groups/yu3grp/projects/scRNASeq/yu3grp/hzhou98/PublicDataSets/ImageNet21K/fall11"
TREE_PATH="./resources/imagenet21k_miil_tree.pth"

# Model configuration
MODEL_NAME="tresnet_m"  # Options: tresnet_m, tresnet_l, resnet50, mobilenetv3_large_100, vit_base_patch16_224
MODEL_PATH="./pretrained_models/${MODEL_NAME}_1k.pth"  # ImageNet-1K pretrained weights

# Training hyperparameters
BATCH_SIZE=64  # Per GPU batch size
EPOCHS=80
LR=3e-4
WEIGHT_DECAY=1e-4
LABEL_SMOOTH=0.2
NUM_WORKERS=8
IMAGE_SIZE=224
NUM_CLASSES=11221

# Output paths
OUTPUT_DIR="./output/${MODEL_NAME}_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${OUTPUT_DIR}/logs"

# ========================================
# Environment Setup
# ========================================

# Create output directories
mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}
mkdir -p logs

# Load required modules (adjust based on your cluster configuration)
module load cuda11.4
module load conda3/202011
conda activate imagenet21k_py308

# Activate virtual environment if needed
# source /path/to/your/venv/bin/activate

# Set environment variables for distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# Get world size (total number of GPUs)
export WORLD_SIZE=$(echo $LSB_HOSTS | wc -w)
export NGPUS_PER_NODE=4  # GPUs per node

# NCCL settings for better performance
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=5

# PyTorch settings
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# ========================================
# Print Configuration
# ========================================

echo "=========================================="
echo "ImageNet-21K Multi-GPU Training"
echo "=========================================="
echo "Job ID: $LSB_JOBID"
echo "Job Name: $LSB_JOBNAME"
echo "Master Node: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "Total Hosts: $LSB_HOSTS"
echo "World Size (Total GPUs): $WORLD_SIZE"
echo "GPUs per Node: $NGPUS_PER_NODE"
echo "Number of Nodes: $((WORLD_SIZE / NGPUS_PER_NODE))"
echo "Batch Size per GPU: $BATCH_SIZE"
echo "Effective Batch Size: $((BATCH_SIZE * WORLD_SIZE))"
echo "Model: $MODEL_NAME"
echo "Training Script: $TRAIN_SCRIPT"
echo "Data Path: $DATA_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo "=========================================="

# Save configuration
cat > ${OUTPUT_DIR}/config.txt <<EOF
Training Configuration
======================
Job ID: $LSB_JOBID
Start Time: $(date)
Model: $MODEL_NAME
Batch Size per GPU: $BATCH_SIZE
Total GPUs: $WORLD_SIZE
Effective Batch Size: $((BATCH_SIZE * WORLD_SIZE))
Learning Rate: $LR
Epochs: $EPOCHS
Data Path: $DATA_PATH
EOF

# ========================================
# Launch Distributed Training
# ========================================

# Function to get GPU rank for each process
get_rank() {
    local node_rank=$1
    local local_rank=$2
    echo $((node_rank * NGPUS_PER_NODE + local_rank))
}

# Parse hosts and launch training on each node
NODE_RANK=0
for HOST in $LSB_HOSTS; do
    # Skip duplicate host entries (LSF lists each slot)
    if [ "$HOST" != "$PREV_HOST" ]; then
        echo "Launching training on node $NODE_RANK: $HOST"
        
        # Launch processes for each GPU on this node
        for ((LOCAL_RANK=0; LOCAL_RANK<$NGPUS_PER_NODE; LOCAL_RANK++)); do
            RANK=$(get_rank $NODE_RANK $LOCAL_RANK)
            
            if [ "$TRAIN_SCRIPT" = "train_semantic_softmax.py" ]; then
                ssh $HOST "cd $PWD && \
                    export CUDA_VISIBLE_DEVICES=$LOCAL_RANK && \
                    export MASTER_ADDR=$MASTER_ADDR && \
                    export MASTER_PORT=$MASTER_PORT && \
                    export WORLD_SIZE=$WORLD_SIZE && \
                    export RANK=$RANK && \
                    python -u $TRAIN_SCRIPT \
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
                        --local_rank=$LOCAL_RANK \
                        2>&1 | tee ${LOG_DIR}/node_${NODE_RANK}_gpu_${LOCAL_RANK}.log" &
            else
                ssh $HOST "cd $PWD && \
                    export CUDA_VISIBLE_DEVICES=$LOCAL_RANK && \
                    export MASTER_ADDR=$MASTER_ADDR && \
                    export MASTER_PORT=$MASTER_PORT && \
                    export WORLD_SIZE=$WORLD_SIZE && \
                    export RANK=$RANK && \
                    python -u $TRAIN_SCRIPT \
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
                        --local_rank=$LOCAL_RANK \
                        2>&1 | tee ${LOG_DIR}/node_${NODE_RANK}_gpu_${LOCAL_RANK}.log" &
            fi
        done
        
        NODE_RANK=$((NODE_RANK + 1))
        PREV_HOST=$HOST
    fi
done

# Wait for all background processes to complete
wait

# ========================================
# Post-training
# ========================================

echo "=========================================="
echo "Training completed at: $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="

# Save completion status
echo "End Time: $(date)" >> ${OUTPUT_DIR}/config.txt
echo "Status: Completed" >> ${OUTPUT_DIR}/config.txt

exit 0
