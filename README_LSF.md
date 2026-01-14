# LSF Training Scripts Summary

This document provides an overview of all LSF training scripts created for ImageNet-21K multi-GPU training.

## 📁 Files Created

### 1. Core Training Scripts

#### `lsf_train_single_node.sh` ⭐ **RECOMMENDED FOR BEGINNERS**
- **Purpose**: Single-node multi-GPU training
- **Configuration**: 1 node × 4 GPUs (default)
- **Use Case**: Testing, debugging, small-scale experiments
- **Complexity**: ★☆☆☆☆ (Easiest)
- **Launch Method**: `torch.distributed.launch`

#### `lsf_train_multi_gpu.sh`
- **Purpose**: Multi-node multi-GPU training with manual control
- **Configuration**: 4 nodes × 4 GPUs = 16 GPUs (default)
- **Use Case**: Production training, large-scale experiments
- **Complexity**: ★★★☆☆ (Advanced)
- **Launch Method**: Manual SSH + process spawning

#### `lsf_train_multinode.sh`
- **Purpose**: Multi-node training using jsrun or mpirun
- **Configuration**: Flexible nodes × GPUs
- **Use Case**: HPC clusters with job launchers
- **Complexity**: ★★★★☆ (Advanced)
- **Launch Method**: jsrun or mpirun

### 2. Setup and Utility Scripts

#### `setup_training.sh` 🛠️ **RUN THIS FIRST**
- **Purpose**: One-time environment setup
- **Features**:
  - Downloads semantic trees
  - Downloads pretrained ImageNet-1K models
  - Verifies Python dependencies
  - Checks data availability
  - Creates directory structure
- **Usage**: `./setup_training.sh`

#### `submit_job.sh` 🚀 **INTERACTIVE JOB SUBMISSION**
- **Purpose**: Interactive helper for submitting jobs
- **Features**:
  - Guided model selection
  - Automatic configuration generation
  - Custom script creation
  - Job submission
- **Usage**: `./submit_job.sh`

### 3. Documentation

#### `LSF_TRAINING_GUIDE.md` 📖 **COMPREHENSIVE GUIDE**
- Complete documentation
- Detailed configuration options
- Troubleshooting guide
- Performance tuning tips
- Expected results and benchmarks

#### `QUICK_START.md` ⚡ **QUICK REFERENCE**
- 5-minute quick start
- Common commands
- Configuration templates
- Troubleshooting cheat sheet

#### `README_LSF.md` (this file)
- Overview of all scripts
- Decision tree for script selection
- Common workflows

## 🎯 Which Script Should I Use?

### Decision Tree

```
START HERE
│
├─ First time user?
│  └─ YES → Use setup_training.sh first
│     └─ Then use submit_job.sh (interactive)
│        OR lsf_train_single_node.sh
│
├─ Need to test quickly?
│  └─ Use lsf_train_single_node.sh (4 GPUs, ~40h)
│
├─ Production training?
│  ├─ Cluster has jsrun/mpirun?
│  │  └─ YES → lsf_train_multinode.sh
│  │  └─ NO → lsf_train_multi_gpu.sh
│  │
│  └─ How many GPUs?
│     ├─ 4-8 GPUs → lsf_train_single_node.sh
│     └─ 16+ GPUs → lsf_train_multi_gpu.sh or multinode
│
└─ Want interactive setup?
   └─ Use submit_job.sh
```

## 🚀 Common Workflows

### Workflow 1: First-Time User (Recommended)

```bash
# Step 1: Setup environment (once)
./setup_training.sh

# Step 2: Use interactive submission
./submit_job.sh
# Follow prompts to configure and submit

# Step 3: Monitor
bjobs
tail -f logs/train_*.out
```

### Workflow 2: Quick Experiment

```bash
# Step 1: Edit configuration
nano lsf_train_single_node.sh
# Modify: MODEL_NAME, BATCH_SIZE, EPOCHS (maybe set to 1 for testing)

# Step 2: Submit
bsub < lsf_train_single_node.sh

# Step 3: Monitor
bjobs -w
tail -f logs/train_*.out
```

### Workflow 3: Production Training

```bash
# Step 1: Verify setup
./setup_training.sh  # If not done already

# Step 2: Choose and edit script
nano lsf_train_multi_gpu.sh
# Configure nodes, GPUs, model, batch size

# Step 3: Test with 1 epoch
# Temporarily set EPOCHS=1 to verify

# Step 4: Full training
# Restore EPOCHS=80 and submit
bsub < lsf_train_multi_gpu.sh

# Step 5: Long-term monitoring
# Check periodically over 1-3 days
bjobs
tail logs/train_*.out
```

## 📊 Script Comparison Matrix

| Feature | single_node.sh | multi_gpu.sh | multinode.sh | submit_job.sh |
|---------|----------------|--------------|--------------|---------------|
| **Complexity** | ★☆☆☆☆ | ★★★☆☆ | ★★★★☆ | ★☆☆☆☆ |
| **Max GPUs** | 8 | Unlimited | Unlimited | Configurable |
| **Setup Time** | 2 min | 5 min | 5 min | 1 min |
| **Flexibility** | Low | High | Medium | High |
| **Launch Method** | torch.distributed | SSH + manual | jsrun/mpirun | Custom |
| **Best For** | Testing | Production | HPC | Beginners |
| **Interactive** | No | No | No | Yes |

## 🔧 Key Configuration Parameters

All scripts share these common parameters:

### Model Configuration
```bash
MODEL_NAME="tresnet_m"     # Model architecture
MODEL_PATH="./pretrained_models/tresnet_m_1k.pth"  # Pretrained weights
```

### Training Hyperparameters
```bash
BATCH_SIZE=64              # Per-GPU batch size
EPOCHS=80                  # Training epochs
LR=3e-4                    # Learning rate
WEIGHT_DECAY=1e-4          # Weight decay
LABEL_SMOOTH=0.2           # Label smoothing
NUM_WORKERS=8              # Data loading workers
IMAGE_SIZE=224             # Input image size
NUM_CLASSES=11221          # Number of classes
```

### Data Paths
```bash
DATA_PATH="/path/to/imagenet21k/fall11"
TREE_PATH="./resources/imagenet21k_miil_tree.pth"  # For semantic softmax
```

### Training Mode
```bash
TRAIN_SCRIPT="train_semantic_softmax.py"  # or "train_single_label.py"
```

## 📈 Expected Performance

### Training Time (80 epochs)

| Model | 4 GPUs | 8 GPUs | 16 GPUs | 32 GPUs |
|-------|--------|--------|---------|---------|
| mobilenetv3_large_100 | 24h | 12h | 6h | 3h |
| resnet50 | 36h | 18h | 9h | 4.5h |
| tresnet_m | 40h | 20h | 10h | 5h |
| tresnet_l | 60h | 30h | 15h | 7.5h |
| vit_base_patch16_224 | 72h | 36h | 18h | 9h |

### Resource Usage (per GPU)

| Model | GPU Memory | Training Speed | Recommended Batch Size |
|-------|------------|----------------|------------------------|
| mobilenetv3_large_100 | 8-10 GB | ~1200 img/s | 128 (V100), 256 (A100) |
| resnet50 | 10-12 GB | ~700 img/s | 64 (V100), 128 (A100) |
| tresnet_m | 11-13 GB | ~650 img/s | 64 (V100), 128 (A100) |
| tresnet_l | 14-16 GB | ~350 img/s | 32 (V100), 64 (A100) |
| vit_base_patch16_224 | 13-15 GB | ~340 img/s | 32 (V100), 64 (A100) |

## 🐛 Troubleshooting

### Script Selection Issues

**Q: Which script should I use for 4 GPUs?**
- A: Use `lsf_train_single_node.sh` (simplest)

**Q: I need 16 GPUs, which script?**
- A: Use `lsf_train_multi_gpu.sh` or `lsf_train_multinode.sh`

**Q: My cluster doesn't support jsrun/mpirun**
- A: Use `lsf_train_multi_gpu.sh` (manual SSH method)

### Common Errors

**Error: "No module named 'timm'"**
```bash
# Solution:
pip install timm
# or
pip install -r requirements.txt
```

**Error: "NCCL error: unhandled system error"**
```bash
# Solution: Check NCCL network settings in script
export NCCL_SOCKET_IFNAME=ib0  # or eth0, or your network interface
export NCCL_DEBUG=INFO
```

**Error: "CUDA out of memory"**
```bash
# Solution: Reduce batch size in script
BATCH_SIZE=32  # or 16
```

**Error: "Cannot find data path"**
```bash
# Solution: Verify and update DATA_PATH in script
ls -la /path/to/imagenet21k/fall11
```

## 📚 Additional Resources

### Documentation
- `LSF_TRAINING_GUIDE.md` - Comprehensive guide
- `QUICK_START.md` - Quick reference
- Original paper: https://arxiv.org/abs/2104.10972
- GitHub repo: https://github.com/Alibaba-MIIL/ImageNet21K

### Support
1. Check documentation first
2. Review LSF cluster documentation
3. Contact HPC support for cluster-specific issues

## ✅ Pre-Flight Checklist

Before submitting any job:

- [ ] Ran `./setup_training.sh`
- [ ] Downloaded semantic tree (for semantic softmax)
- [ ] Downloaded pretrained model (optional but recommended)
- [ ] Verified data path exists
- [ ] Created `logs/` directory
- [ ] Edited script configuration
- [ ] Tested with EPOCHS=1 (optional but recommended)
- [ ] Have ~100GB disk space for outputs
- [ ] Understand expected training time

## 🎓 Learning Path

1. **Beginner**: Start with `lsf_train_single_node.sh`
   - Use 1 node, 4 GPUs
   - Set EPOCHS=1 for initial test
   - Monitor logs and understand output

2. **Intermediate**: Try different models
   - Experiment with MODEL_NAME
   - Adjust BATCH_SIZE for your GPUs
   - Compare training times and accuracy

3. **Advanced**: Scale to multiple nodes
   - Use `lsf_train_multi_gpu.sh`
   - Configure 16+ GPUs
   - Optimize hyperparameters

4. **Expert**: Production deployment
   - Fine-tune for your cluster
   - Implement checkpointing
   - Custom loss functions and metrics

## 📝 Quick Command Reference

```bash
# Setup
./setup_training.sh              # One-time setup

# Submit jobs
bsub < lsf_train_single_node.sh  # Single node
./submit_job.sh                  # Interactive
bsub < lsf_train_multi_gpu.sh    # Multi-node

# Monitor
bjobs                            # List jobs
bjobs -w                         # Wide format
bjobs -l <JOBID>                 # Detailed info
bhist <JOBID>                    # Job history
tail -f logs/train_*.out         # View logs

# Manage
bkill <JOBID>                    # Kill job
bstop <JOBID>                    # Suspend job
bresume <JOBID>                  # Resume job
```

---

## 🚀 Ready to Start?

```bash
# Quick start in 3 commands:
./setup_training.sh              # Setup
./submit_job.sh                  # Submit (interactive)
tail -f logs/train_*.out         # Monitor
```

**Good luck with your training!** 🎉
