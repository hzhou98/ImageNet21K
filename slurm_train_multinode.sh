#!/bin/bash
# --------------------------------------------------------
# Slurm Multi-Node Multi-GPU Training Script
# For AI/ML Training Clusters
# --------------------------------------------------------

#SBATCH --job-name=imagenet21k_multinode
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=72:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --partition=gpu

# Description:
# Multi-node distributed training for ImageNet-21K
# Default: 4 nodes × 4 GPUs = 16 GPUs
# Adjust --nodes and --gres=gpu:X as needed

# ========================================
# Configuration
# ========================================

TRAIN_SCRIPT="train_semantic_softmax.py"
DATA_PATH="/research/groups/yu3grp/projects/scRNASeq/yu3grp/hzhou98/PublicDataSets/ImageNet21K/imagenet21k_resized_new"
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

NNODES=${SLURM_NNODES}
NGPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-4}
WORLD_SIZE=$((NNODES * NGPUS_PER_NODE))

OUTPUT_DIR="./output/${MODEL_NAME}_$(date +%Y%m%d_%H%M%S)"

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
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ib0

# Get master node address
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

export MASTER_ADDR
export MASTER_PORT

# ========================================
# Info
# ========================================

echo "=========================================="
echo "ImageNet-21K Multi-Node Training (Slurm)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "Number of Nodes: $NNODES"
echo "GPUs per Node: $NGPUS_PER_NODE"
echo "Total GPUs: $WORLD_SIZE"
echo "Master Node: $MASTER_ADDR:$MASTER_PORT"
echo "Model: $MODEL_NAME"
echo "Batch Size per GPU: $BATCH_SIZE"
echo "Effective Batch Size: $((BATCH_SIZE * WORLD_SIZE))"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

cat > ${OUTPUT_DIR}/config.txt <<EOF
Training Configuration
======================
Job ID: $SLURM_JOB_ID
Nodes: $SLURM_NODELIST
Start Time: $(date)
Number of Nodes: $NNODES
GPUs per Node: $NGPUS_PER_NODE
Total GPUs: $WORLD_SIZE
Model: $MODEL_NAME
Batch Size per GPU: $BATCH_SIZE
Effective Batch Size: $((BATCH_SIZE * WORLD_SIZE))
Learning Rate: $LR
Epochs: $EPOCHS
Data Path: $DATA_PATH
EOF

# ========================================
# Launch Training with srun
# ========================================

if [ "$TRAIN_SCRIPT" = "train_semantic_softmax.py" ]; then
    srun python -m torch.distributed.launch \
        --nproc_per_node=$NGPUS_PER_NODE \
        --nnodes=$NNODES \
        --node_rank=\$SLURM_NODEID \
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
        2>&1 | tee ${OUTPUT_DIR}/training.log
else
    srun python -m torch.distributed.launch \
        --nproc_per_node=$NGPUS_PER_NODE \
        --nnodes=$NNODES \
        --node_rank=\$SLURM_NODEID \
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
        2>&1 | tee ${OUTPUT_DIR}/training.log
fi

echo "Training completed at: $(date)"
echo "End Time: $(date)" >> ${OUTPUT_DIR}/config.txt
echo "Status: Completed" >> ${OUTPUT_DIR}/config.txt
