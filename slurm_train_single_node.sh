#!/bin/bash
# --------------------------------------------------------
# Slurm Single-Node Multi-GPU Training Script
# For AI/ML Training Clusters
# --------------------------------------------------------

#SBATCH --job-name=imagenet21k_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --partition=gpu

# Description:
# Single-node multi-GPU training for ImageNet-21K
# Default: 1 node with 4 GPUs
# Adjust --gres=gpu:X for different GPU counts

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
NGPUS=${SLURM_GPUS_ON_NODE:-4}

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

# Or activate conda environment
# source /path/to/conda/bin/activate
# conda activate pytorch_env

# Environment variables
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NCCL_DEBUG=INFO

# Set master address for distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# ========================================
# Info
# ========================================

echo "=========================================="
echo "ImageNet-21K Single-Node Training (Slurm)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $NGPUS"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Model: $MODEL_NAME"
echo "Batch Size per GPU: $BATCH_SIZE"
echo "Effective Batch Size: $((BATCH_SIZE * NGPUS))"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Save configuration
cat > ${OUTPUT_DIR}/config.txt <<EOF
Training Configuration
======================
Job ID: $SLURM_JOB_ID
Node: $SLURM_NODELIST
Start Time: $(date)
Model: $MODEL_NAME
GPUs: $NGPUS
Batch Size per GPU: $BATCH_SIZE
Effective Batch Size: $((BATCH_SIZE * NGPUS))
Learning Rate: $LR
Epochs: $EPOCHS
Data Path: $DATA_PATH
EOF

# ========================================
# Launch Training
# ========================================

if [ "$TRAIN_SCRIPT" = "train_semantic_softmax.py" ]; then
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
        2>&1 | tee ${OUTPUT_DIR}/training.log
fi

# ========================================
# Post-training
# ========================================

echo "=========================================="
echo "Training completed at: $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="

echo "End Time: $(date)" >> ${OUTPUT_DIR}/config.txt
echo "Status: Completed" >> ${OUTPUT_DIR}/config.txt

exit 0
