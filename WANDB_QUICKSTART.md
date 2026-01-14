# Weights & Biases Integration - Quick Reference

## 📊 What's New

Weights & Biases (W&B) integration added for real-time training monitoring and visualization.

## 🆕 New Files Created

### Training Scripts (W&B-Enabled)
- `train_single_label_wandb.py` - Single-label training with W&B tracking
- `train_semantic_softmax_wandb.py` - Semantic softmax training with W&B tracking

### LSF Scripts
- `lsf_train_wandb.sh` - LSF batch script with W&B support

### Utilities
- `setup_wandb.sh` - Interactive W&B setup script

### Documentation
- `WANDB_GUIDE.md` - Comprehensive W&B guide
- `WANDB_QUICKSTART.md` - This file

## ⚡ Quick Start (3 Steps)

### Step 1: Setup W&B
```bash
./setup_wandb.sh
```
This will:
- Install wandb (if needed)
- Configure W&B login
- Set up your W&B entity and project
- Update LSF scripts

### Step 2: Submit Training Job
```bash
bsub < lsf_train_wandb.sh
```

### Step 3: View Dashboard
Check job output for W&B URL:
```bash
tail -f logs/train_*.out | grep "wandb.ai"
```

Click the URL to view your live training dashboard!

## 📈 What Gets Tracked

### Real-Time Metrics
- **Training Loss** - Updated every 100 batches
- **Learning Rate** - Current LR at each step
- **Validation Accuracy** - Top-1 and Top-5 accuracy per epoch
- **Training Speed** - Images per second
- **GPU Utilization** - Memory and compute usage
- **System Stats** - CPU, memory, disk I/O

### Hyperparameters
All training configurations automatically logged:
- Model architecture, batch size, learning rate
- Number of GPUs, effective batch size
- Data augmentation settings
- And more...

## 🎨 Dashboard Features

### Live Training Plots
- Loss curves (training and validation)
- Accuracy trends over time
- Learning rate schedule
- Training throughput

### System Monitoring
- GPU memory usage
- GPU utilization percentage
- CPU and system memory
- Network I/O

### Experiment Comparison
- Compare multiple runs side-by-side
- Parallel coordinates plot
- Table view with sortable columns

## 🔧 Configuration

### Enable/Disable W&B
Edit `lsf_train_wandb.sh`:
```bash
# Enable W&B
USE_WANDB=true

# Disable W&B (falls back to standard training)
USE_WANDB=false
```

### Project Settings
```bash
WANDB_PROJECT="imagenet21k-training"
WANDB_ENTITY="your_username"  # Optional
WANDB_RUN_NAME="custom_name"  # Auto-generated if empty
```

### Choose Training Script
```bash
# Semantic Softmax (recommended)
TRAIN_SCRIPT="train_semantic_softmax_wandb.py"

# Single Label
TRAIN_SCRIPT="train_single_label_wandb.py"
```

## 📖 Common Workflows

### Basic Training with W&B
```bash
# 1. Setup (one-time)
./setup_wandb.sh

# 2. Submit job
bsub < lsf_train_wandb.sh

# 3. Monitor
tail -f logs/train_*.out
```

### Offline Training (No Internet)
Edit `lsf_train_wandb.sh`:
```bash
export WANDB_MODE=offline
```

After training, sync to cloud:
```bash
wandb sync output/tresnet_m_*/wandb/
```

### Compare Multiple Experiments
```bash
# Run 1: Baseline
WANDB_RUN_NAME="baseline" bsub < lsf_train_wandb.sh

# Run 2: Higher LR (edit LR in script)
WANDB_RUN_NAME="high_lr" bsub < lsf_train_wandb.sh

# View comparison at:
# https://wandb.ai/username/project
```

## 🐛 Troubleshooting

### "wandb not installed"
```bash
pip install wandb
```

### "API key not configured"
```bash
wandb login
# Or
export WANDB_API_KEY="your_key"
```

### "Failed to connect"
```bash
# Use offline mode
export WANDB_MODE=offline
```

### Slow training with W&B
```bash
# Disable system stats
export WANDB_DISABLE_SYSTEM_STATS=true

# Or disable model watching
# Comment out wandb.watch() in training script
```

## 📊 Expected Overhead

- **Negligible**: < 1% training time with default settings
- **Logging frequency**: Every 100 batches (adjustable)
- **Model watching**: Optional, adds ~5% overhead

## 🎯 Best Practices

### 1. Use Descriptive Run Names
```bash
WANDB_RUN_NAME="tresnet_m_64bs_3e-4lr_$(date +%Y%m%d)"
```

### 2. Organize by Project
```bash
# Different projects for different experiments
WANDB_PROJECT="imagenet21k-baseline"
WANDB_PROJECT="imagenet21k-ablation"
```

### 3. Tag Your Runs
Training scripts automatically add tags, or add custom ones:
```python
wandb.init(tags=["tresnet_m", "semantic_softmax", "experiment1"])
```

### 4. Add Notes
Document what makes each run special:
```python
wandb.config.update({"notes": "Testing new augmentation strategy"})
```

## 🔗 Quick Links

- **W&B Homepage**: https://wandb.ai
- **Your Runs**: https://wandb.ai/username/imagenet21k-training
- **Documentation**: https://docs.wandb.ai
- **Get API Key**: https://wandb.ai/authorize

## 📚 Documentation

- **Comprehensive Guide**: See `WANDB_GUIDE.md`
- **Training Scripts**: `train_*_wandb.py`
- **LSF Scripts**: `lsf_train_wandb.sh`

## ✅ Comparison: With vs Without W&B

### Without W&B (Original)
```bash
# Submit job
bsub < lsf_train_single_node.sh

# Monitor (terminal only)
tail -f logs/train_*.out

# Results: Local log files only
```

### With W&B (New)
```bash
# Submit job
bsub < lsf_train_wandb.sh

# Monitor (web dashboard)
# Open browser to W&B URL

# Results: 
# - Live dashboard
# - All metrics saved to cloud
# - Easy comparison between runs
# - Share results with team
```

## 🎓 Learning Resources

### Video Tutorials
- Getting Started: https://www.youtube.com/wandb
- Advanced Features: https://wandb.ai/site/tutorials

### Example Projects
- W&B Gallery: https://wandb.ai/gallery
- PyTorch Examples: https://github.com/wandb/examples

## 💡 Pro Tips

1. **Bookmark your dashboard** for quick access during training
2. **Set up mobile app** for monitoring on the go
3. **Create custom dashboards** for specific metrics you care about
4. **Use tags** to organize related experiments
5. **Share run links** with your team for collaboration

## 🎉 Summary

**To start using W&B:**

```bash
# One-time setup
./setup_wandb.sh

# Use W&B-enabled training
bsub < lsf_train_wandb.sh

# View dashboard
# Check job output for URL
```

**Benefits:**
- ✅ Real-time visualization
- ✅ No manual logging needed
- ✅ Compare experiments easily
- ✅ Share results with team
- ✅ Access from anywhere

**Dashboard**: https://wandb.ai

Happy tracking! 🚀

---

For detailed information, see [WANDB_GUIDE.md](WANDB_GUIDE.md)
