# Cluster Training Scripts - Quick Reference

This project includes training scripts for both **LSF** and **Slurm** job schedulers.

## 🎯 Which Scripts Should I Use?

### LSF Cluster (IBM Spectrum LSF)
If your cluster uses **LSF** commands (`bsub`, `bjobs`, `bkill`):
- ✅ Use `lsf_train_*.sh` scripts
- ✅ See [LSF_TRAINING_GUIDE.md](LSF_TRAINING_GUIDE.md)

### Slurm Cluster
If your cluster uses **Slurm** commands (`sbatch`, `squeue`, `scancel`):
- ✅ Use `slurm_train_*.sh` scripts
- ✅ See [SLURM_GUIDE.md](SLURM_GUIDE.md)

### Not Sure?
```bash
# Check which system you have
which bsub    # LSF
which sbatch  # Slurm
```

## 📁 Available Scripts

### LSF Scripts
```
lsf_train_single_node.sh      # Single node (1×4 GPUs)
lsf_train_multi_gpu.sh        # Multi-node (4×4 GPUs)
lsf_train_multinode.sh        # Multi-node with jsrun
lsf_train_wandb.sh            # Single node + W&B
```

### Slurm Scripts
```
slurm_train_single_node.sh    # Single node (1×4 GPUs)
slurm_train_multinode.sh      # Multi-node (4×4 GPUs)
slurm_train_wandb.sh          # Single node + W&B
```

## ⚡ Quick Start

### LSF
```bash
# Submit job
bsub < lsf_train_single_node.sh

# Monitor
bjobs
tail -f logs/train_*.out
```

### Slurm
```bash
# Submit job
sbatch slurm_train_single_node.sh

# Monitor
squeue -u $USER
tail -f logs/train_*.out
```

## 🎨 With W&B Tracking

### LSF
```bash
./setup_wandb.sh
bsub < lsf_train_wandb.sh
```

### Slurm
```bash
./setup_wandb.sh
sbatch slurm_train_wandb.sh
```

## 📊 Command Comparison

| Action | LSF | Slurm |
|--------|-----|-------|
| **Submit** | `bsub < script.sh` | `sbatch script.sh` |
| **List jobs** | `bjobs` | `squeue -u $USER` |
| **Cancel** | `bkill JOBID` | `scancel JOBID` |
| **Details** | `bjobs -l JOBID` | `scontrol show job JOBID` |
| **History** | `bhist JOBID` | `sacct -j JOBID` |

## 📖 Documentation

### General Guides
- [QUICK_START.md](QUICK_START.md) - Quick reference for all systems
- [WORKFLOW.md](WORKFLOW.md) - Complete training workflow

### LSF-Specific
- [LSF_TRAINING_GUIDE.md](LSF_TRAINING_GUIDE.md) - Comprehensive LSF guide
- [README_LSF.md](README_LSF.md) - LSF scripts overview

### Slurm-Specific
- [SLURM_GUIDE.md](SLURM_GUIDE.md) - Comprehensive Slurm guide

### W&B Integration
- [WANDB_QUICKSTART.md](WANDB_QUICKSTART.md) - Quick W&B guide
- [WANDB_GUIDE.md](WANDB_GUIDE.md) - Comprehensive W&B docs
- [WANDB_SUMMARY.md](WANDB_SUMMARY.md) - W&B integration summary

## 🔧 Configuration

Both LSF and Slurm scripts share the same configuration parameters:

```bash
MODEL_NAME="tresnet_m"        # Model architecture
BATCH_SIZE=64                 # Per-GPU batch size
EPOCHS=80                     # Training epochs
DATA_PATH="/path/to/data"     # Dataset location
TRAIN_SCRIPT="train_semantic_softmax.py"  # Training mode
```

## 🎓 Examples

### Single Node Training (4 GPUs)

**LSF:**
```bash
# Edit configuration
nano lsf_train_single_node.sh

# Submit
bsub < lsf_train_single_node.sh
```

**Slurm:**
```bash
# Edit configuration
nano slurm_train_single_node.sh

# Submit
sbatch slurm_train_single_node.sh
```

### Multi-Node Training (16 GPUs)

**LSF:**
```bash
bsub < lsf_train_multi_gpu.sh
```

**Slurm:**
```bash
sbatch slurm_train_multinode.sh
```

### With W&B Tracking

**LSF:**
```bash
./setup_wandb.sh
bsub < lsf_train_wandb.sh
```

**Slurm:**
```bash
./setup_wandb.sh
sbatch slurm_train_wandb.sh
```

## 🚀 Recommended First Run

### For LSF Users
```bash
# 1. Setup
./setup_training.sh

# 2. Test run (optional)
# Edit: Set EPOCHS=1
# nano lsf_train_single_node.sh

# 3. Submit
bsub < lsf_train_single_node.sh

# 4. Monitor
bjobs
tail -f logs/train_*.out
```

### For Slurm Users
```bash
# 1. Setup
./setup_training.sh

# 2. Test run (optional)
# Edit: Set EPOCHS=1
# nano slurm_train_single_node.sh

# 3. Submit
sbatch slurm_train_single_node.sh

# 4. Monitor
squeue -u $USER
tail -f logs/train_*.out
```

## 📈 Expected Performance

Same performance across both systems:
- **TResNet-M, 4 GPUs**: ~40 hours for 80 epochs
- **TResNet-M, 16 GPUs**: ~10 hours for 80 epochs
- **Training speed**: ~3000 img/sec (4 GPUs)

## 🐛 Getting Help

### LSF Issues
- See [LSF_TRAINING_GUIDE.md](LSF_TRAINING_GUIDE.md) troubleshooting section
- Contact your HPC LSF support team

### Slurm Issues
- See [SLURM_GUIDE.md](SLURM_GUIDE.md) troubleshooting section
- Contact your HPC Slurm support team

### Training Issues
- Check [QUICK_START.md](QUICK_START.md)
- Review training logs in `logs/` directory

## ✅ File Overview

### Training Scripts (Python)
- `train_single_label.py` - Single-label training
- `train_semantic_softmax.py` - Semantic softmax training
- `train_*_wandb.py` - W&B-enabled versions

### Job Scripts
- **LSF**: `lsf_train_*.sh` (4 scripts)
- **Slurm**: `slurm_train_*.sh` (3 scripts)

### Utilities
- `setup_training.sh` - Environment setup
- `setup_wandb.sh` - W&B setup
- `verify_setup.sh` - Pre-flight checks
- `submit_job.sh` - Interactive job submission (LSF)

### Documentation
- `QUICK_START.md` - Quick reference
- `LSF_TRAINING_GUIDE.md` - LSF comprehensive guide
- `SLURM_GUIDE.md` - Slurm comprehensive guide
- `WANDB_GUIDE.md` - W&B integration guide
- `WORKFLOW.md` - Complete workflow

## 🎉 Summary

**For LSF clusters:**
```bash
bsub < lsf_train_single_node.sh
```

**For Slurm clusters:**
```bash
sbatch slurm_train_single_node.sh
```

**With W&B (both):**
```bash
./setup_wandb.sh
# Then use *_wandb.sh script
```

---

**Choose your system and get started!** 🚀

For detailed instructions:
- **LSF**: [LSF_TRAINING_GUIDE.md](LSF_TRAINING_GUIDE.md)
- **Slurm**: [SLURM_GUIDE.md](SLURM_GUIDE.md)
