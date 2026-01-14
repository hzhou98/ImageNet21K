# ImageNet-21K LSF Training - Quick Reference

## File Overview

```
ImageNet21K/
├── lsf_train_single_node.sh    # Single node, multi-GPU (RECOMMENDED for getting started)
├── lsf_train_multi_gpu.sh      # Multi-node, advanced control
├── lsf_train_multinode.sh      # Multi-node with jsrun/mpirun
├── setup_training.sh           # Setup script (run first!)
├── LSF_TRAINING_GUIDE.md       # Detailed documentation
└── QUICK_START.md              # This file
```

## Quick Start (5 minutes)

### 1. Setup Environment
```bash
# Run setup script
./setup_training.sh

# Follow prompts to download:
# - Semantic tree (required for semantic softmax)
# - Pretrained models (optional but recommended)
```

### 2. Edit Configuration
Open `lsf_train_single_node.sh` and modify:

```bash
# Essential parameters to check/modify:
MODEL_NAME="tresnet_m"          # Your model choice
BATCH_SIZE=64                   # Adjust based on GPU memory
DATA_PATH="/your/path/to/data"  # Your dataset path

# Optional: Switch training mode
TRAIN_SCRIPT="train_semantic_softmax.py"  # or "train_single_label.py"
```

### 3. Submit Job
```bash
# Create logs directory (if not exists)
mkdir -p logs

# Submit
bsub < lsf_train_single_node.sh

# Check status
bjobs
```

### 4. Monitor Training
```bash
# View output
tail -f logs/train_<JOBID>.out

# Check job details
bjobs -l <JOBID>

# See GPU usage
bjobs -gpu <JOBID>
```

## Common Commands

### Job Management
```bash
# Submit job
bsub < lsf_train_single_node.sh

# List jobs
bjobs

# Detailed info
bjobs -l <JOBID>

# Kill job
bkill <JOBID>

# Job history
bhist <JOBID>
```

### Log Monitoring
```bash
# Real-time output
tail -f logs/train_<JOBID>.out

# Error log
tail -f logs/train_<JOBID>.err

# Last 100 lines
tail -n 100 logs/train_<JOBID>.out

# Search for errors
grep -i error logs/train_<JOBID>.err
```

## Model Selection

| Use Case | Model | Batch Size (V100) | Training Time |
|----------|-------|-------------------|---------------|
| **Fast experiments** | mobilenetv3_large_100 | 128 | ~24h (4 GPUs) |
| **Baseline** | resnet50 | 64 | ~36h (4 GPUs) |
| **Recommended** | tresnet_m | 64 | ~40h (4 GPUs) |
| **High accuracy** | tresnet_l | 32 | ~60h (4 GPUs) |
| **Transformer** | vit_base_patch16_224 | 32 | ~72h (4 GPUs) |

## Batch Size Guidelines

**V100 16GB:**
- tresnet_m: 64
- tresnet_l: 32
- resnet50: 64

**A100 40GB:**
- tresnet_m: 128
- tresnet_l: 64
- resnet50: 128

**A100 80GB:**
- tresnet_m: 256
- tresnet_l: 128
- resnet50: 256

## Training Scripts Comparison

### Single Node (Recommended for Beginners)
```bash
# File: lsf_train_single_node.sh
# - Simplest setup
# - 1 node, 4 GPUs
# - Uses torch.distributed.launch
# - Best for: experiments, debugging
```

### Multi Node (Advanced)
```bash
# File: lsf_train_multi_gpu.sh
# - Manual process spawning
# - Configurable nodes × GPUs
# - Full control over distribution
# - Best for: production, large-scale
```

### Multi Node with jsrun
```bash
# File: lsf_train_multinode.sh
# - Uses jsrun or mpirun
# - Automatic GPU detection
# - Most scalable
# - Best for: HPC clusters
```

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
BATCH_SIZE=32  # or 16
```

### NCCL Errors
```bash
# Check network interface in script
export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=INFO
```

### Slow Training
```bash
# Increase data loading workers
NUM_WORKERS=16

# Check data is on fast storage
# Consider copying to local SSD
```

### Module Not Found
```bash
# Install requirements
pip install -r requirements.txt

# Or individually
pip install torch torchvision timm Pillow
```

### Cannot Find Data
```bash
# Verify data path
ls -la /path/to/imagenet21k/fall11

# Check number of classes
find /path/to/imagenet21k/fall11 -mindepth 1 -maxdepth 1 -type d | wc -l
# Should be ~11221 classes
```

## Configuration Templates

### Semantic Softmax (Recommended)
```bash
TRAIN_SCRIPT="train_semantic_softmax.py"
TREE_PATH="./resources/imagenet21k_miil_tree.pth"
LABEL_SMOOTH=0.2
```

### Single Label
```bash
TRAIN_SCRIPT="train_single_label.py"
LABEL_SMOOTH=0.2
# No tree_path needed
```

### Fast Convergence
```bash
EPOCHS=80
LR=3e-4
WEIGHT_DECAY=1e-4
# Initialize from ImageNet-1K
```

### From Scratch
```bash
EPOCHS=140
LR=0.1
WEIGHT_DECAY=1e-4
MODEL_PATH=""  # Empty for random init
```

## Expected Performance (TResNet-M, 4 GPUs)

### Training Metrics
- Epoch time: ~25-30 minutes
- Images/sec: ~3000-3500
- Total time: ~30-40 hours
- GPU memory: ~12GB/GPU

### Accuracy Milestones
- Epoch 20: ~72% top-1
- Epoch 40: ~75% top-1
- Epoch 60: ~76% top-1
- Epoch 80: ~76.4% top-1

## Next Steps After Training

### 1. Find Your Model
```bash
cd output/
ls -lt | head  # Find latest run
```

### 2. Transfer to ImageNet-1K
```bash
# Fine-tune on ImageNet-1K for 20 epochs
python train_single_label.py \
    --data_path=/path/to/imagenet1k \
    --model_path=output/tresnet_m_*/checkpoint.pth \
    --num_classes=1000 \
    --epochs=20 \
    --lr=1e-4
```

### 3. Use for Downstream Tasks
```python
import torch
from src_files.models import create_model

# Load your trained model
model = create_model(args)
checkpoint = torch.load('output/tresnet_m_*/checkpoint.pth')
model.load_state_dict(checkpoint)

# Fine-tune on your task
```

## Tips for Faster Training

1. **Use mixed precision** (already enabled)
2. **Optimal batch size**: Fill GPU memory to ~90%
3. **Fast data loading**: Use SSD/NVMe storage
4. **More workers**: NUM_WORKERS=16 per GPU
5. **Multi-node**: Scale to 16+ GPUs
6. **Pre-download**: Cache dataset to local storage

## Need Help?

1. **Detailed docs**: See `LSF_TRAINING_GUIDE.md`
2. **Original repo**: https://github.com/Alibaba-MIIL/ImageNet21K
3. **Paper**: https://arxiv.org/abs/2104.10972
4. **Cluster help**: Contact your HPC support team

## Common Workflow

```bash
# 1. Setup (once)
./setup_training.sh

# 2. Edit config
nano lsf_train_single_node.sh

# 3. Test with small run (optional)
# Set EPOCHS=1 for testing

# 4. Submit full training
bsub < lsf_train_single_node.sh

# 5. Monitor
bjobs
tail -f logs/train_*.out

# 6. Wait 1-3 days...

# 7. Check results
cd output/tresnet_m_*/
cat config.txt
```

---

**Ready to train?** Run `./setup_training.sh` to begin!
