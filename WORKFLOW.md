# ImageNet-21K LSF Training - Complete Workflow

## 📋 Files Created

### Training Scripts (LSF Batch Jobs)
```
lsf_train_single_node.sh    ⭐ Single node, 4-8 GPUs (recommended start)
lsf_train_multi_gpu.sh         Multi-node with manual control  
lsf_train_multinode.sh         Multi-node with jsrun/mpirun
```

### Utility Scripts
```
setup_training.sh           🛠️ Environment setup (run first!)
submit_job.sh              🚀 Interactive job submission
verify_setup.sh            ✅ Pre-flight verification
```

### Documentation
```
LSF_TRAINING_GUIDE.md      📖 Comprehensive guide
QUICK_START.md             ⚡ Quick reference
README_LSF.md              📋 Scripts overview
WORKFLOW.md                🔄 This file
```

## 🔄 Complete Workflow

### Step 1: Initial Setup (One-Time)
```bash
# Run setup script
./setup_training.sh

# This will:
# ✓ Create directories (logs, resources, pretrained_models, output)
# ✓ Download semantic tree
# ✓ Download pretrained ImageNet-1K models (optional)
# ✓ Check Python environment
# ✓ Verify dependencies
```

### Step 2: Verify Environment
```bash
# Run verification script
./verify_setup.sh

# This checks:
# ✓ Directory structure
# ✓ Training scripts
# ✓ Semantic tree
# ✓ Pretrained models
# ✓ Python packages (torch, timm, etc.)
# ✓ CUDA availability
# ✓ Data path
# ✓ LSF environment
# ✓ Disk space
# ✓ Documentation
```

### Step 3: Choose Your Path

#### Path A: Interactive (Easiest) 🌟
```bash
./submit_job.sh

# Interactive prompts:
# 1. Select model (mobilenetv3, resnet50, tresnet_m, tresnet_l, vit)
# 2. Select training mode (semantic softmax or single label)
# 3. Select GPU configuration (4, 8, 16, or 32 GPUs)
# 4. Confirm batch size
# 5. Set epochs
# 6. Verify data path
# 7. Name your job
# 8. Auto-generates and submits custom script
```

#### Path B: Manual (More Control) 🎛️
```bash
# 1. Choose a script based on your needs:
#    - Single node: lsf_train_single_node.sh
#    - Multi-node: lsf_train_multi_gpu.sh

# 2. Edit the script
nano lsf_train_single_node.sh

# 3. Configure key parameters:
MODEL_NAME="tresnet_m"
BATCH_SIZE=64
EPOCHS=80
DATA_PATH="/your/path"
TRAIN_SCRIPT="train_semantic_softmax.py"

# 4. Submit
bsub < lsf_train_single_node.sh
```

### Step 4: Monitor Training
```bash
# Check job status
bjobs
bjobs -w          # Wide format
bjobs -l <JOBID>  # Detailed info

# Monitor logs in real-time
tail -f logs/train_<JOBID>.out

# Check for errors
tail -f logs/train_<JOBID>.err

# View GPU usage
bjobs -gpu <JOBID>

# Job history
bhist <JOBID>
```

### Step 5: Manage Job (If Needed)
```bash
# Suspend job
bstop <JOBID>

# Resume job
bresume <JOBID>

# Kill job
bkill <JOBID>

# Resubmit if needed
bsub < your_script.sh
```

### Step 6: Collect Results
```bash
# Navigate to output directory
cd output/

# Find your run
ls -lt | head

# Check results
cd tresnet_m_20260114_120000/
cat config.txt
cat training.log
```

## 📊 Decision Tree

```
┌─────────────────────────────────────┐
│     Start Here: Which to use?       │
└──────────────┬──────────────────────┘
               │
               ├─── First time? ────────> ./submit_job.sh (interactive)
               │                              ↓
               │                         Auto-configured
               │
               ├─── Quick test? ───────> lsf_train_single_node.sh
               │                         (4 GPUs, simple)
               │
               ├─── Need 16+ GPUs? ────> lsf_train_multi_gpu.sh
               │                         or lsf_train_multinode.sh
               │
               └─── Production? ───────> Depends on cluster:
                                         • Has jsrun? → multinode.sh
                                         • No jsrun?  → multi_gpu.sh
```

## 🎯 Common Scenarios

### Scenario 1: I want to quickly test with TResNet-M
```bash
# Fast path (2 minutes)
./submit_job.sh
# Select: 3 (tresnet_m), 1 (semantic softmax), 1 (4 GPUs)
# Accept defaults, submit
```

### Scenario 2: Production training, 16 GPUs, TResNet-L
```bash
# 1. Edit multi-GPU script
nano lsf_train_multi_gpu.sh

# 2. Set configuration:
MODEL_NAME="tresnet_l"
BATCH_SIZE=32
EPOCHS=80
# Verify: 4 nodes × 4 GPUs = 16 GPUs in LSF headers

# 3. Test with 1 epoch first
EPOCHS=1
bsub < lsf_train_multi_gpu.sh

# 4. If successful, run full training
EPOCHS=80
bsub < lsf_train_multi_gpu.sh
```

### Scenario 3: I want to experiment with different models
```bash
# Use single node script for quick iterations
nano lsf_train_single_node.sh

# Try different models:
# Run 1: MODEL_NAME="mobilenetv3_large_100"
# Run 2: MODEL_NAME="resnet50"  
# Run 3: MODEL_NAME="tresnet_m"

# Compare results in output/ directory
```

### Scenario 4: Training from scratch (no pretrained weights)
```bash
# Edit script
nano lsf_train_single_node.sh

# Modify:
MODEL_PATH=""     # Empty = random initialization
EPOCHS=140        # More epochs needed
LR=0.1           # Higher learning rate for scratch
```

### Scenario 5: Fine-tune for downstream task
```bash
# After ImageNet-21K pretraining
MODEL_PATH="output/tresnet_m_date/checkpoint.pth"
NUM_CLASSES=1000  # Or your task's number of classes
EPOCHS=20         # Fewer epochs for fine-tuning
LR=1e-4          # Lower learning rate
```

## 🔍 Monitoring Checklist

During training, check these metrics:

### Every Hour
- [ ] Job still running? (`bjobs`)
- [ ] No errors in log? (`tail logs/train_*.err`)
- [ ] Training progressing? (epoch numbers increasing)

### Every 4-8 Hours
- [ ] Training speed normal? (~3000 img/s for tresnet_m on 4 GPUs)
- [ ] GPU memory stable? (no OOM errors)
- [ ] Accuracy improving? (check validation results)

### Every Day
- [ ] Disk space sufficient? (`df -h`)
- [ ] Expected completion time on track?
- [ ] Compare to expected milestones (see QUICK_START.md)

## 📈 Expected Timeline

### TResNet-M on 4 GPUs (Example)
```
Hour 0:   Job submitted
Hour 1:   Epoch 1-2 complete, ~70% top-1
Hour 12:  Epoch 20 complete, ~72% top-1
Hour 24:  Epoch 40 complete, ~75% top-1
Hour 36:  Epoch 60 complete, ~76% top-1
Hour 40:  Epoch 80 complete, ~76.4% top-1 ✓
```

### Scaling Effects
- **8 GPUs**: ~2x faster (20 hours)
- **16 GPUs**: ~4x faster (10 hours)
- **32 GPUs**: ~8x faster (5 hours)

## 🆘 Troubleshooting Guide

### Job won't submit
```bash
# Check LSF queue
bqueues

# Check your jobs
bjobs -u your_username

# Check LSF resources
bhosts

# Verify script syntax
bash -n lsf_train_single_node.sh
```

### Job fails immediately
```bash
# Check error log
cat logs/train_<JOBID>.err

# Common fixes:
# - Module not loaded: Check module load commands
# - Path wrong: Verify DATA_PATH exists
# - CUDA error: Check GPU allocation in LSF headers
```

### Out of memory during training
```bash
# Reduce batch size in script
BATCH_SIZE=32  # or 16

# Or reduce image size
IMAGE_SIZE=192  # instead of 224
```

### Slow training speed
```bash
# Check data loading
NUM_WORKERS=16  # Increase workers

# Check I/O
# - Is data on fast storage (SSD/NVMe)?
# - Consider copying to local /tmp/

# Check GPU utilization
nvidia-smi  # Should be ~90-100%
```

### NCCL communication errors
```bash
# In script, adjust NCCL settings:
export NCCL_SOCKET_IFNAME=ib0  # or eth0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800
```

## ✅ Success Indicators

You know training is successful when:

1. **Job runs without errors** for full duration
2. **Training speed** matches expectations (~3000 img/s)
3. **GPU memory** stable (~12GB for tresnet_m)
4. **Validation accuracy** improves each epoch
5. **Final accuracy** matches paper (~76.4% for tresnet_m)
6. **Output files** created in output/ directory

## 🎓 Next Steps After Training

### 1. Evaluate Your Model
```bash
cd output/tresnet_m_*/
cat training.log | grep "Acc_Top1"
```

### 2. Transfer to ImageNet-1K
```bash
python train_single_label.py \
    --data_path=/path/to/imagenet1k \
    --model_path=output/tresnet_m_*/checkpoint.pth \
    --num_classes=1000 \
    --epochs=20
```

### 3. Use for Downstream Tasks
- Object detection
- Semantic segmentation  
- Classification on custom datasets
- See Transfer_learning.md

### 4. Share/Archive Results
```bash
# Create archive
tar -czf tresnet_m_imagenet21k.tar.gz output/tresnet_m_*/

# Document your settings
cp output/tresnet_m_*/config.txt experiments_log.txt
```

## 📚 Quick Reference

### Most Common Commands
```bash
# Setup and verify
./setup_training.sh
./verify_setup.sh

# Submit (interactive)
./submit_job.sh

# Submit (manual)
bsub < lsf_train_single_node.sh

# Monitor
bjobs
tail -f logs/train_*.out

# Control
bkill <JOBID>
```

### Key Files to Check
```bash
# Job output
logs/train_<JOBID>.out

# Job errors  
logs/train_<JOBID>.err

# Training config
output/model_name_*/config.txt

# Training log
output/model_name_*/training.log
```

## 🎉 You're Ready!

**Recommended first run:**
```bash
# 1. Setup
./setup_training.sh

# 2. Verify
./verify_setup.sh

# 3. Submit
./submit_job.sh
# Choose: tresnet_m, semantic softmax, 4 GPUs

# 4. Monitor
bjobs
tail -f logs/train_*.out

# 5. Wait ~40 hours...

# 6. Celebrate! 🎉
```

---

**Questions?** See:
- `QUICK_START.md` - Quick reference
- `LSF_TRAINING_GUIDE.md` - Detailed guide  
- `README_LSF.md` - Scripts overview
