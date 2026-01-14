# LSF Cluster Training Guide for ImageNet-21K

This directory contains LSF batch scripts for distributed training on GPU clusters using PyTorch DistributedDataParallel.

## Available Scripts

### 1. `lsf_train_single_node.sh` - Single Node Multi-GPU
**Best for:** Quick experiments, small models, or limited cluster access
- Runs on 1 node with multiple GPUs (default: 4 GPUs)
- Simplest setup with `torch.distributed.launch`
- Typical training time: 2-3 days for 80 epochs

**Usage:**
```bash
# Edit configuration in the script, then submit
bsub < lsf_train_single_node.sh

# Monitor job
bjobs
bhist -l <JOBID>

# View logs
tail -f logs/train_<JOBID>.out
```

### 2. `lsf_train_multi_gpu.sh` - Multi-Node Multi-GPU (Advanced)
**Best for:** Large-scale training, faster convergence, production runs
- Runs across multiple nodes (default: 4 nodes × 4 GPUs = 16 GPUs)
- Manual process spawning for maximum control
- Typical training time: 12-18 hours for 80 epochs

### 3. `lsf_train_multinode.sh` - Multi-Node with jsrun/mpirun
**Best for:** HPC clusters with jsrun or MPI support
- Supports both IBM jsrun and standard MPI launchers
- Automatic GPU detection and distribution
- Most scalable option

## Quick Start

### Step 1: Prepare Prerequisites

1. **Download pretrained ImageNet-1K weights** (recommended for faster convergence):
```bash
mkdir -p pretrained_models
cd pretrained_models

# Example: Download TResNet-M weights from timm
python -c "import timm; model = timm.create_model('tresnet_m', pretrained=True); import torch; torch.save(model.state_dict(), 'tresnet_m_1k.pth')"
```

2. **Download semantic tree** (for semantic softmax training):
```bash
mkdir -p resources
cd resources
wget https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/resources/fall11/imagenet21k_miil_tree.pth
```

3. **Verify data path** exists:
```bash
ls /research/groups/yu3grp/projects/scRNASeq/yu3grp/hzhou98/PublicDataSets/ImageNet21K/fall11
```

### Step 2: Configure Training Parameters

Edit the script and adjust these key parameters:

```bash
# Model Selection
MODEL_NAME="tresnet_m"  # Options: tresnet_m, tresnet_l, resnet50, mobilenetv3_large_100

# Training Settings
BATCH_SIZE=64           # Per-GPU batch size (adjust based on GPU memory)
EPOCHS=80               # 80 epochs recommended for fall11, 40 for winter21
LR=3e-4                 # Learning rate
NUM_CLASSES=11221       # 11221 for fall11, 10450 for winter21

# Data Path
DATA_PATH="/path/to/your/imagenet21k/fall11"

# Script Selection
TRAIN_SCRIPT="train_semantic_softmax.py"  # or "train_single_label.py"
```

### Step 3: Submit Job

```bash
# Create logs directory
mkdir -p logs

# Submit job
bsub < lsf_train_single_node.sh

# Check job status
bjobs -w

# View output in real-time
tail -f logs/train_<JOBID>.out
```

## Configuration Guide

### GPU and Memory Settings

**For V100 16GB GPUs:**
```bash
BATCH_SIZE=64    # For tresnet_m
BATCH_SIZE=32    # For tresnet_l or vit_base
```

**For A100 40GB GPUs:**
```bash
BATCH_SIZE=128   # For tresnet_m
BATCH_SIZE=64    # For tresnet_l or vit_base
```

**For A100 80GB GPUs:**
```bash
BATCH_SIZE=256   # For tresnet_m
BATCH_SIZE=128   # For tresnet_l
```

### LSF Resource Requests

**Single node, 4 GPUs:**
```bash
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=4:mode=exclusive_process"
```

**Multiple nodes (4 nodes × 4 GPUs):**
```bash
#BSUB -n 32                    # 4 nodes × 8 cores
#BSUB -R "span[ptile=8]"       # 8 cores per node
#BSUB -gpu "num=4:mode=exclusive_process"
```

**Using nnodes (newer LSF):**
```bash
#BSUB -nnodes 4
#BSUB -alloc_flags gpumps
```

### Training Modes

**1. Semantic Softmax (Recommended):**
```bash
TRAIN_SCRIPT="train_semantic_softmax.py"
# Requires --tree_path parameter
# Better accuracy, especially for hierarchical classification
```

**2. Single Label:**
```bash
TRAIN_SCRIPT="train_single_label.py"
# Standard cross-entropy loss
# Simpler, no semantic tree needed
```

## Model Selection Guide

| Model | Params | Top-1 (21K) | Top-1 (1K) | Batch Size (V100) | Training Time |
|-------|--------|-------------|------------|-------------------|---------------|
| mobilenetv3_large_100 | 5.5M | 73.1% | 78.0% | 128 | ~24h (4 GPUs) |
| resnet50 | 25.6M | 75.6% | 82.0% | 64 | ~36h (4 GPUs) |
| tresnet_m | 31.4M | 76.4% | 83.1% | 64 | ~40h (4 GPUs) |
| tresnet_l | 55.9M | 76.7% | 83.9% | 32 | ~60h (4 GPUs) |
| vit_base_patch16_224 | 86.6M | 77.6% | 84.4% | 32 | ~72h (4 GPUs) |

## Monitoring and Debugging

### Check Job Status
```bash
# List your jobs
bjobs

# Detailed job info
bjobs -l <JOBID>

# Job history
bhist <JOBID>

# GPU usage
bjobs -gpu <JOBID>
```

### View Logs
```bash
# Training output
tail -f logs/train_<JOBID>.out

# Errors
tail -f logs/train_<JOBID>.err

# Per-GPU logs (for multi-node training)
tail -f output/model_name_timestamp/logs/node_0_gpu_0.log
```

### Common Issues

**1. Out of Memory:**
```bash
# Solution: Reduce batch size
BATCH_SIZE=32  # or 16
```

**2. NCCL Timeout:**
```bash
# Solution: Adjust NCCL settings in script
export NCCL_SOCKET_IFNAME=ib0  # Use InfiniBand
export NCCL_IB_DISABLE=0
export NCCL_TIMEOUT=1800
```

**3. Slow Data Loading:**
```bash
# Solution: Increase workers or check I/O
NUM_WORKERS=16  # Increase workers
# Or preload dataset to local SSD
```

**4. Cannot find module:**
```bash
# Solution: Add project to PYTHONPATH
export PYTHONPATH="${PWD}:${PYTHONPATH}"
```

## Performance Tips

### 1. Optimal Batch Size
- Start with recommended batch size
- Increase until GPU memory is ~90% utilized
- Effective batch size = BATCH_SIZE × NUM_GPUS

### 2. Data Loading
- Use `NUM_WORKERS=8-16` per GPU
- Consider using local SSD for faster I/O
- Pin memory for faster GPU transfer

### 3. Mixed Precision
- Already enabled via `torch.cuda.amp`
- Reduces memory by ~40%
- Speeds up training by ~2x

### 4. Learning Rate Scaling
- Linear scaling rule: LR = base_LR × (batch_size / 256)
- For 16 GPUs × 64 batch = 1024 total batch
- Scaled LR = 3e-4 × (1024/256) = 1.2e-3

## Expected Results

### Training Metrics (TResNet-M on 4 GPUs)
- Epoch time: ~25-30 minutes
- Training rate: ~3000-3500 img/sec
- Total training: ~30-40 hours for 80 epochs
- Peak GPU memory: ~12GB per GPU
- Final top-1 accuracy: ~76.4%

### Validation Checkpoints
| Epoch | Top-1 | Top-5 |
|-------|-------|-------|
| 20 | ~72% | ~88% |
| 40 | ~75% | ~90% |
| 60 | ~76% | ~91% |
| 80 | ~76.4% | ~91.5% |

## Post-Training

### Save Model Checkpoint
The trained model will be saved in:
```
output/${MODEL_NAME}_timestamp/
├── config.txt          # Training configuration
├── logs/              # Training logs
└── checkpoints/       # Model checkpoints (if implemented)
```

### Transfer to ImageNet-1K
After pretraining on ImageNet-21K, fine-tune on ImageNet-1K:
```bash
# Use the trained model as initialization
MODEL_PATH="output/tresnet_m_20260114/model_final.pth"

# Fine-tune on ImageNet-1K
python train_single_label.py \
    --data_path=/path/to/imagenet1k \
    --model_path=$MODEL_PATH \
    --num_classes=1000 \
    --epochs=20 \
    --lr=1e-4
```

## Cluster-Specific Adjustments

Different HPC clusters may require specific adjustments:

### For SLURM Clusters
Convert LSF directives to SLURM:
```bash
#SBATCH --job-name=imagenet21k
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00

srun python -m torch.distributed.launch ...
```

### For SGE Clusters
```bash
#$ -N imagenet21k
#$ -pe mpi 16
#$ -l gpu=4

mpirun -np 4 python -m torch.distributed.launch ...
```

## Support

For issues or questions:
1. Check [ImageNet-21K GitHub](https://github.com/Alibaba-MIIL/ImageNet21K)
2. Review LSF cluster documentation
3. Contact your HPC support team for cluster-specific issues

## Citation

If you use this code, please cite:
```bibtex
@misc{ridnik2021imagenet21k,
    title={ImageNet-21K Pretraining for the Masses}, 
    author={Tal Ridnik and Emanuel Ben-Baruch and Asaf Noy and Lihi Zelnik-Manor},
    year={2021},
    eprint={2104.10972}
}
```
