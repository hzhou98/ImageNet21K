#!/bin/bash
# --------------------------------------------------------
# Quick Job Submission Script for ImageNet-21K Training
# Interactive helper for submitting LSF jobs
# --------------------------------------------------------

set -e

echo "=========================================="
echo "ImageNet-21K Training Job Submission"
echo "=========================================="
echo ""

# ========================================
# Configuration Selection
# ========================================

# Model selection
echo "Select model:"
echo "1. mobilenetv3_large_100 (fast, 5.5M params)"
echo "2. resnet50 (baseline, 25M params)"
echo "3. tresnet_m (recommended, 31M params)"
echo "4. tresnet_l (high accuracy, 55M params)"
echo "5. vit_base_patch16_224 (transformer, 86M params)"
read -p "Choice (1-5): " model_choice

case $model_choice in
    1) MODEL_NAME="mobilenetv3_large_100"; BATCH_SIZE=128 ;;
    2) MODEL_NAME="resnet50"; BATCH_SIZE=64 ;;
    3) MODEL_NAME="tresnet_m"; BATCH_SIZE=64 ;;
    4) MODEL_NAME="tresnet_l"; BATCH_SIZE=32 ;;
    5) MODEL_NAME="vit_base_patch16_224"; BATCH_SIZE=32 ;;
    *) echo "Invalid choice"; exit 1 ;;
esac

# Training mode
echo ""
echo "Select training mode:"
echo "1. Semantic Softmax (recommended)"
echo "2. Single Label"
read -p "Choice (1-2): " mode_choice

case $mode_choice in
    1) TRAIN_SCRIPT="train_semantic_softmax.py"; USE_TREE=true ;;
    2) TRAIN_SCRIPT="train_single_label.py"; USE_TREE=false ;;
    *) echo "Invalid choice"; exit 1 ;;
esac

# GPU configuration
echo ""
echo "Select GPU configuration:"
echo "1. Single node, 4 GPUs (recommended for testing)"
echo "2. Single node, 8 GPUs"
echo "3. Multi-node, 4 nodes × 4 GPUs = 16 GPUs"
echo "4. Multi-node, 8 nodes × 4 GPUs = 32 GPUs"
read -p "Choice (1-4): " gpu_choice

case $gpu_choice in
    1) NGPUS=4; NNODES=1; SCRIPT_TYPE="single" ;;
    2) NGPUS=8; NNODES=1; SCRIPT_TYPE="single" ;;
    3) NGPUS=16; NNODES=4; SCRIPT_TYPE="multi" ;;
    4) NGPUS=32; NNODES=8; SCRIPT_TYPE="multi" ;;
    *) echo "Invalid choice"; exit 1 ;;
esac

# Batch size adjustment
echo ""
read -p "Batch size per GPU [default: $BATCH_SIZE]: " user_batch
if [ ! -z "$user_batch" ]; then
    BATCH_SIZE=$user_batch
fi

# Epochs
echo ""
read -p "Number of epochs [default: 80]: " user_epochs
if [ ! -z "$user_epochs" ]; then
    EPOCHS=$user_epochs
else
    EPOCHS=80
fi

# Data path
echo ""
DEFAULT_DATA_PATH="/research/groups/yu3grp/projects/scRNASeq/yu3grp/hzhou98/PublicDataSets/ImageNet21K/fall11"
read -p "Data path [default: $DEFAULT_DATA_PATH]: " user_data
if [ ! -z "$user_data" ]; then
    DATA_PATH=$user_data
else
    DATA_PATH=$DEFAULT_DATA_PATH
fi

# Job name
echo ""
read -p "Job name [default: ${MODEL_NAME}_train]: " user_jobname
if [ ! -z "$user_jobname" ]; then
    JOB_NAME=$user_jobname
else
    JOB_NAME="${MODEL_NAME}_train"
fi

# ========================================
# Summary
# ========================================

echo ""
echo "=========================================="
echo "Configuration Summary"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Training mode: $TRAIN_SCRIPT"
echo "GPUs: $NGPUS ($NNODES nodes)"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Effective batch size: $((BATCH_SIZE * NGPUS))"
echo "Epochs: $EPOCHS"
echo "Data path: $DATA_PATH"
echo "Job name: $JOB_NAME"
echo "=========================================="
echo ""

read -p "Submit job with these settings? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Job submission cancelled"
    exit 0
fi

# ========================================
# Generate Custom Job Script
# ========================================

CUSTOM_SCRIPT="lsf_train_custom_${MODEL_NAME}_$(date +%Y%m%d_%H%M%S).sh"

echo "Generating custom job script: $CUSTOM_SCRIPT"

cat > $CUSTOM_SCRIPT << EOF
#!/bin/bash
#BSUB -P ImageNet21K_Training
#BSUB -J $JOB_NAME
#BSUB -o logs/train_%J.out
#BSUB -e logs/train_%J.err
#BSUB -W 72:00
EOF

if [ $NNODES -eq 1 ]; then
    cat >> $CUSTOM_SCRIPT << EOF
#BSUB -n $NGPUS
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=$NGPUS:mode=exclusive_process:mps=no:j_exclusive=yes"
EOF
else
    cat >> $CUSTOM_SCRIPT << EOF
#BSUB -n $((NGPUS * 8 / 4))
#BSUB -R "span[ptile=8]"
#BSUB -gpu "num=4:mode=exclusive_process:mps=no:j_exclusive=yes"
EOF
fi

cat >> $CUSTOM_SCRIPT << EOF
#BSUB -q gpu
#BSUB -R "rusage[mem=32000]"

# Auto-generated configuration
TRAIN_SCRIPT="$TRAIN_SCRIPT"
DATA_PATH="$DATA_PATH"
MODEL_NAME="$MODEL_NAME"
MODEL_PATH="./pretrained_models/\${MODEL_NAME}_1k.pth"
BATCH_SIZE=$BATCH_SIZE
EPOCHS=$EPOCHS
LR=3e-4
WEIGHT_DECAY=1e-4
LABEL_SMOOTH=0.2
NUM_WORKERS=8
IMAGE_SIZE=224
NUM_CLASSES=11221
OUTPUT_DIR="./output/\${MODEL_NAME}_\$(date +%Y%m%d_%H%M%S)"

mkdir -p \${OUTPUT_DIR}
mkdir -p logs

module load cuda/11.7
module load gcc/9.3.0
module load python/3.9

export OMP_NUM_THREADS=4
export NCCL_DEBUG=INFO

echo "=========================================="
echo "Training Configuration"
echo "=========================================="
echo "Model: \$MODEL_NAME"
echo "GPUs: $NGPUS"
echo "Batch size per GPU: \$BATCH_SIZE"
echo "Effective batch size: $((BATCH_SIZE * NGPUS))"
echo "Epochs: \$EPOCHS"
echo "Output: \$OUTPUT_DIR"
echo "=========================================="

EOF

if [ $NNODES -eq 1 ]; then
    # Single node
    if [ "$USE_TREE" = true ]; then
        cat >> $CUSTOM_SCRIPT << 'EOF'
python -m torch.distributed.launch \
    --nproc_per_node=$NGPUS \
    --master_port=29500 \
    $TRAIN_SCRIPT \
    --data_path=$DATA_PATH \
    --tree_path=./resources/imagenet21k_miil_tree.pth \
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
EOF
    else
        cat >> $CUSTOM_SCRIPT << 'EOF'
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
EOF
    fi
else
    # Multi-node
    cat >> $CUSTOM_SCRIPT << 'EOF'
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=$LSB_DJOB_NUMPROC

# Launch on each node
for HOST in $(echo $LSB_HOSTS | tr ' ' '\n' | sort -u); do
    ssh $HOST "cd $PWD && \
        python -m torch.distributed.launch \
        --nproc_per_node=4 \
        --nnodes=$NNODES \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        $TRAIN_SCRIPT \
        --data_path=$DATA_PATH \
        --model_name=$MODEL_NAME \
        --model_path=$MODEL_PATH \
        --batch_size=$BATCH_SIZE \
        --epochs=$EPOCHS \
        2>&1 | tee ${OUTPUT_DIR}/training_${HOST}.log" &
done
wait
EOF
fi

cat >> $CUSTOM_SCRIPT << EOF

echo "Training completed at: \$(date)"
echo "Output: \$OUTPUT_DIR"
EOF

chmod +x $CUSTOM_SCRIPT

# ========================================
# Submit Job
# ========================================

echo "Submitting job..."
bsub < $CUSTOM_SCRIPT

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Job submitted successfully!"
    echo "=========================================="
    echo "Custom script saved: $CUSTOM_SCRIPT"
    echo ""
    echo "Monitor your job:"
    echo "  bjobs"
    echo "  tail -f logs/train_<JOBID>.out"
    echo ""
    echo "Expected training time:"
    case $MODEL_NAME in
        "mobilenetv3_large_100") echo "  ~24 hours on 4 GPUs" ;;
        "resnet50") echo "  ~36 hours on 4 GPUs" ;;
        "tresnet_m") echo "  ~40 hours on 4 GPUs" ;;
        "tresnet_l") echo "  ~60 hours on 4 GPUs" ;;
        "vit_base_patch16_224") echo "  ~72 hours on 4 GPUs" ;;
    esac
    echo ""
    echo "Scaling factor: $(bc <<< "scale=1; 4.0/$NGPUS"x faster with $NGPUS GPUs"
    echo ""
else
    echo "Job submission failed"
    exit 1
fi
