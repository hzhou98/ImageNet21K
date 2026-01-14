# Slurm Training Guide for ImageNet-21K

Complete guide for training ImageNet-21K models on Slurm AI/ML clusters.

## 📁 Slurm Scripts Created

### Single Node Training
- **[slurm_train_single_node.sh](slurm_train_single_node.sh)** - 1 node, multi-GPU (recommended start)
- **[slurm_train_wandb.sh](slurm_train_wandb.sh)** - 1 node with W&B tracking

### Multi-Node Training
- **[slurm_train_multinode.sh](slurm_train_multinode.sh)** - Multi-node distributed training

## 🚀 Quick Start

### Step 1: Setup Environment
```bash
# Run setup script (if not already done)
./setup_training.sh
```

### Step 2: Choose Your Script

**For beginners / testing:**
```bash
sbatch slurm_train_single_node.sh
```

**For tracking with W&B:**
```bash
# Setup W&B first
./setup_wandb.sh

# Submit job
sbatch slurm_train_wandb.sh
```

**For large-scale training:**
```bash
sbatch slurm_train_multinode.sh
```

### Step 3: Monitor Job
```bash
# Check job status
squeue -u $USER

# View output in real-time
tail -f logs/train_*.out

# Check job details
scontrol show job <JOBID>
```

## 📊 Script Comparison

| Script | Nodes | GPUs | Use Case | W&B |
|--------|-------|------|----------|-----|
| slurm_train_single_node.sh | 1 | 4 | Testing, small runs | ❌ |
| slurm_train_wandb.sh | 1 | 4 | Tracking experiments | ✅ |
| slurm_train_multinode.sh | 4+ | 16+ | Production, fast training | ❌ |

## ⚙️ Configuration

### Resource Allocation

**Single Node (4 GPUs):**
```bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
```

**Multi-Node (4 nodes × 4 GPUs = 16 GPUs):**
```bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
```

**GPU Types (if needed):**
```bash
#SBATCH --gres=gpu:v100:4      # V100 GPUs
#SBATCH --gres=gpu:a100:4      # A100 GPUs
#SBATCH --gres=gpu:h100:4      # H100 GPUs
```

### Batch Size Guidelines

**V100 16GB:**
```bash
BATCH_SIZE=64    # tresnet_m
BATCH_SIZE=32    # tresnet_l
```

**A100 40GB:**
```bash
BATCH_SIZE=128   # tresnet_m
BATCH_SIZE=64    # tresnet_l
```

**A100 80GB:**
```bash
BATCH_SIZE=256   # tresnet_m
BATCH_SIZE=128   # tresnet_l
```

### Time Limits

```bash
#SBATCH --time=48:00:00    # 48 hours for single node
#SBATCH --time=72:00:00    # 72 hours for multi-node
```

### Partition Selection

```bash
#SBATCH --partition=gpu           # General GPU partition
#SBATCH --partition=gpu-a100      # A100-specific
#SBATCH --partition=gpu-preempt   # Preemptible (cheaper)
```

## 🎯 Common Workflows

### Workflow 1: Quick Test Run
```bash
# Edit script: Set EPOCHS=1 for testing
nano slurm_train_single_node.sh

# Submit
sbatch slurm_train_single_node.sh

# Monitor
squeue -u $USER
tail -f logs/train_*.out
```

### Workflow 2: Production Training with W&B
```bash
# Setup W&B (one-time)
./setup_wandb.sh

# Edit configuration
nano slurm_train_wandb.sh
# Set: MODEL_NAME, BATCH_SIZE, EPOCHS

# Submit
sbatch slurm_train_wandb.sh

# View dashboard
# Check output for W&B URL
```

### Workflow 3: Large-Scale Multi-Node
```bash
# Edit for more nodes/GPUs
nano slurm_train_multinode.sh
# Adjust: --nodes=8 for 32 GPUs

# Submit
sbatch slurm_train_multinode.sh

# Monitor on all nodes
tail -f output/*/training.log
```

## 🔧 Slurm Commands Reference

### Job Submission
```bash
# Submit job
sbatch script.sh

# Submit with custom job name
sbatch --job-name=my_experiment script.sh

# Submit with email notification
sbatch --mail-type=END,FAIL --mail-user=your@email.com script.sh
```

### Job Monitoring
```bash
# List your jobs
squeue -u $USER

# Detailed job info
scontrol show job <JOBID>

# Job efficiency stats
seff <JOBID>

# Job accounting info
sacct -j <JOBID> --format=JobID,JobName,Partition,Account,AllocCPUS,State,ExitCode
```

### Job Control
```bash
# Cancel job
scancel <JOBID>

# Cancel all your jobs
scancel -u $USER

# Hold job (prevent from running)
scontrol hold <JOBID>

# Release held job
scontrol release <JOBID>
```

### Cluster Information
```bash
# View partitions
sinfo

# View available nodes
sinfo -N

# Check node details
scontrol show node <NODENAME>

# Check GPU availability
sinfo -o "%20N %10c %10m %25f %10G"
```

## 📈 Expected Performance

### Training Time (TResNet-M, 80 epochs)

| GPUs | Configuration | Time | Speedup |
|------|---------------|------|---------|
| 4 | 1 node | ~40h | 1x |
| 8 | 1 node | ~20h | 2x |
| 16 | 4 nodes | ~10h | 4x |
| 32 | 8 nodes | ~5h | 8x |

### Resource Requirements (per GPU)

| Model | GPU Memory | Batch Size (V100) | CPUs | Memory |
|-------|------------|-------------------|------|--------|
| mobilenetv3 | 8-10 GB | 128 | 4-8 | 32 GB |
| resnet50 | 10-12 GB | 64 | 4-8 | 32 GB |
| tresnet_m | 11-13 GB | 64 | 4-8 | 32 GB |
| tresnet_l | 14-16 GB | 32 | 4-8 | 32 GB |
| vit_base | 13-15 GB | 32 | 4-8 | 32 GB |

## 🐛 Troubleshooting

### Job Pending
```bash
# Check reason
squeue -u $USER --start

# Common reasons:
# - Resources: Waiting for available GPUs
# - Priority: Other jobs have higher priority
# - QOSMaxJobsPerUserLimit: Too many jobs running
```

### Out of Memory
```bash
# Solution 1: Reduce batch size
BATCH_SIZE=32  # or 16

# Solution 2: Request more memory
#SBATCH --mem=256G

# Solution 3: Reduce image size
IMAGE_SIZE=192
```

### NCCL Errors
```bash
# Check network settings in script
export NCCL_SOCKET_IFNAME=ib0  # or eth0
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
```

### Module Not Found
```bash
# Check available modules
module avail

# Load correct versions
module load cuda/11.7
module load python/3.9

# Or use conda
source /path/to/conda/bin/activate
conda activate pytorch_env
```

### GPU Not Detected
```bash
# Check GPU allocation
srun --jobid=$SLURM_JOB_ID nvidia-smi

# Verify CUDA is working
srun --jobid=$SLURM_JOB_ID python -c "import torch; print(torch.cuda.is_available())"
```

## 🎓 Advanced Features

### Interactive Session
```bash
# Request interactive GPU node
srun --nodes=1 --gres=gpu:1 --time=2:00:00 --pty bash

# Run commands interactively
nvidia-smi
python train_single_label.py --epochs=1 --use_wandb
```

### Array Jobs (Multiple Experiments)
```bash
# Create array job script
#SBATCH --array=1-3

# Different configs per job
BATCH_SIZES=(32 64 128)
BATCH_SIZE=${BATCH_SIZES[$SLURM_ARRAY_TASK_ID-1]}

# Submit array job
sbatch script.sh
```

### Checkpoint and Resume
```python
# Add to training script
if args.resume:
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
```

### Job Dependencies
```bash
# Submit job that waits for another
JOBID1=$(sbatch script1.sh | awk '{print $4}')
sbatch --dependency=afterok:$JOBID1 script2.sh
```

## 📊 Monitoring Tips

### Real-Time GPU Monitoring
```bash
# On running job
srun --jobid=<JOBID> nvidia-smi

# Continuous monitoring
watch -n 1 'squeue -u $USER'
```

### Log Files
```bash
# Training output
tail -f logs/train_<JOBID>.out

# Error log
tail -f logs/train_<JOBID>.err

# Follow both
tail -f logs/train_<JOBID>.{out,err}
```

### W&B Dashboard
```bash
# Get W&B URL from logs
grep "wandb.ai" logs/train_*.out

# Monitor in browser
# Open the URL for live metrics
```

## 🔄 Slurm vs LSF Comparison

| Feature | Slurm | LSF |
|---------|-------|-----|
| Submit | `sbatch` | `bsub <` |
| Queue | `squeue` | `bjobs` |
| Cancel | `scancel` | `bkill` |
| Info | `scontrol show job` | `bjobs -l` |
| Nodes | `#SBATCH --nodes=4` | `#BSUB -n 32` |
| GPUs | `#SBATCH --gres=gpu:4` | `#BSUB -gpu "num=4"` |
| Time | `#SBATCH --time=48:00:00` | `#BSUB -W 48:00` |
| Output | `#SBATCH --output=file` | `#BSUB -o file` |

## ✅ Best Practices

### 1. Test Before Full Run
```bash
# Always test with 1-2 epochs first
EPOCHS=1
sbatch slurm_train_single_node.sh
```

### 2. Use Job Names
```bash
#SBATCH --job-name=tresnet_m_exp1
```

### 3. Set Email Notifications
```bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your@email.com
```

### 4. Save Checkpoints
```bash
# Add checkpoint saving in training script
# Every N epochs, save model state
```

### 5. Monitor Resource Usage
```bash
# After job completes
seff <JOBID>
```

## 📚 Additional Resources

### Slurm Documentation
- Official Docs: https://slurm.schedmd.com
- Cheat Sheet: https://slurm.schedmd.com/pdfs/summary.pdf
- Tutorials: https://slurm.schedmd.com/tutorials.html

### Training Scripts
- Single-label: `train_single_label.py`
- Semantic softmax: `train_semantic_softmax.py`
- With W&B: `*_wandb.py` versions

### Setup Scripts
- Environment: `./setup_training.sh`
- W&B: `./setup_wandb.sh`
- Verification: `./verify_setup.sh`

## 🎉 Summary

**Slurm scripts created:**
- ✅ `slurm_train_single_node.sh` - Single node training
- ✅ `slurm_train_wandb.sh` - With W&B tracking
- ✅ `slurm_train_multinode.sh` - Multi-node training

**Quick start:**
```bash
# 1. Submit job
sbatch slurm_train_single_node.sh

# 2. Monitor
squeue -u $USER
tail -f logs/train_*.out

# 3. Check results
ls output/
```

**For W&B tracking:**
```bash
./setup_wandb.sh
sbatch slurm_train_wandb.sh
```

---

For more details, see:
- [LSF_TRAINING_GUIDE.md](LSF_TRAINING_GUIDE.md) - Similar concepts for LSF
- [WANDB_GUIDE.md](WANDB_GUIDE.md) - W&B integration details
- [QUICK_START.md](QUICK_START.md) - General training guide
