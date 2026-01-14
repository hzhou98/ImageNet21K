# Weights & Biases Integration Guide for ImageNet-21K Training

This guide explains how to use Weights & Biases (W&B) to track and visualize your ImageNet-21K training experiments with a live dashboard.

## 🎯 What is Weights & Biases?

Weights & Biases is a platform for tracking machine learning experiments. It provides:

- **Live Training Dashboard**: Real-time loss, accuracy, and training metrics
- **System Monitoring**: GPU utilization, memory usage, CPU metrics
- **Experiment Comparison**: Compare multiple runs side-by-side
- **Model Artifacts**: Save and version model checkpoints
- **Hyperparameter Tracking**: Automatic logging of all configurations
- **Collaboration**: Share results with your team

**Dashboard URL**: https://wandb.ai

## 🚀 Quick Start (3 Steps)

### Step 1: Install W&B
```bash
pip install wandb
```

### Step 2: Login (One-Time Setup)
```bash
# Get your API key from: https://wandb.ai/authorize
wandb login

# Or set environment variable
export WANDB_API_KEY="your_api_key_here"
```

### Step 3: Run Training with W&B
```bash
# Edit and submit W&B-enabled training script
bsub < lsf_train_wandb.sh

# Or manually add --use_wandb flag
python train_semantic_softmax_wandb.py \
    --data_path=/path/to/data \
    --model_name=tresnet_m \
    --use_wandb \
    --wandb_project=imagenet21k \
    --wandb_run_name=tresnet_m_experiment1
```

## 📁 Files Created for W&B Integration

### Training Scripts (W&B-Enabled)
```
train_single_label_wandb.py         # Single-label training with W&B
train_semantic_softmax_wandb.py     # Semantic softmax training with W&B
```

### LSF Script
```
lsf_train_wandb.sh                  # LSF batch script with W&B support
```

## 📊 Tracked Metrics

### Training Metrics (Per Batch)
- `train/batch_loss` - Loss per batch (every 100 batches)
- `train/learning_rate` - Current learning rate
- `train/epoch` - Current epoch
- `train/batch` - Global batch number

### Training Metrics (Per Epoch)
- `train/epoch_loss` - Average epoch loss
- `train/images_per_sec` - Training throughput
- `train/learning_rate_epoch` - End-of-epoch learning rate

### Validation Metrics
- `val/top1_accuracy` - Top-1 validation accuracy
- `val/top5_accuracy` - Top-5 validation accuracy
- `val/semantic_top1_accuracy` - Semantic top-1 accuracy (semantic softmax only)

### Hyperparameters (Automatically Logged)
- Model name, learning rate, batch size, epochs
- Image size, number of classes, weight decay
- Number of GPUs, effective batch size
- Training mode (single-label or semantic)

### System Metrics (Optional)
- GPU utilization, memory, temperature
- CPU utilization and memory
- Disk I/O

## 🎨 Dashboard Features

### 1. Real-Time Plots
Watch your metrics update live during training:
- Loss curves (training and validation)
- Accuracy curves (top-1, top-5)
- Learning rate schedule
- Training speed (img/sec)

### 2. System Monitoring
Track hardware utilization:
- GPU memory usage
- GPU utilization %
- CPU and system memory
- Network I/O

### 3. Model Watching (Optional)
Track model internals:
- Gradient distributions
- Parameter distributions
- Layer-wise statistics

### 4. Run Comparison
Compare multiple experiments:
- Side-by-side metric plots
- Parallel coordinates for hyperparameters
- Table view with sortable columns

## ⚙️ Configuration Options

### Basic Configuration
```bash
# Enable W&B
--use_wandb

# Set project name
--wandb_project=imagenet21k

# Set run name (auto-generated if not specified)
--wandb_run_name=tresnet_m_exp1

# Set entity (username or team)
--wandb_entity=your_username
```

### Environment Variables
```bash
# API key
export WANDB_API_KEY="your_api_key"

# Mode: online, offline, disabled
export WANDB_MODE=online

# Cache and data directory
export WANDB_DIR=/path/to/output
export WANDB_CACHE_DIR=/path/to/cache

# Disable system stats (reduces overhead)
export WANDB_DISABLE_SYSTEM_STATS=true

# Silent mode (less console output)
export WANDB_SILENT=true
```

## 🔧 LSF Script Configuration

Edit `lsf_train_wandb.sh`:

```bash
# Enable/disable W&B
USE_WANDB=true

# W&B project settings
WANDB_PROJECT="imagenet21k-training"
WANDB_ENTITY="your_username"  # Optional
WANDB_RUN_NAME="custom_run_name"  # Auto-generated if empty

# Choose training script (W&B-enabled)
TRAIN_SCRIPT="train_semantic_softmax_wandb.py"
# or
TRAIN_SCRIPT="train_single_label_wandb.py"
```

## 📖 Example Workflows

### Workflow 1: Basic Training with W&B
```bash
# 1. Login once
wandb login

# 2. Edit LSF script
nano lsf_train_wandb.sh
# Set: USE_WANDB=true
# Set: WANDB_PROJECT="my_project"

# 3. Submit job
bsub < lsf_train_wandb.sh

# 4. View dashboard
# Check job output for W&B URL
tail -f logs/train_*.out
# Look for: "Weights & Biases initialized: https://wandb.ai/..."
```

### Workflow 2: Offline Logging (No Internet)
```bash
# In LSF script, set offline mode
export WANDB_MODE=offline

# Submit job
bsub < lsf_train_wandb.sh

# After training, sync to cloud
wandb sync output/tresnet_m_*/wandb/
```

### Workflow 3: Comparing Multiple Runs
```bash
# Run 1: Baseline
wandb_run_name="baseline_tresnet_m" bsub < lsf_train_wandb.sh

# Run 2: Higher learning rate
# Edit script: LR=5e-4
wandb_run_name="high_lr_tresnet_m" bsub < lsf_train_wandb.sh

# Run 3: Larger batch size
# Edit script: BATCH_SIZE=128
wandb_run_name="large_batch_tresnet_m" bsub < lsf_train_wandb.sh

# Compare on W&B dashboard
# https://wandb.ai/your_username/imagenet21k-training
```

### Workflow 4: Resume Failed Run
```bash
# W&B automatically handles resuming if run_id is set
# Add to your training script or LSF script:

# Option 1: In LSF script before python command
export WANDB_RESUME=allow
export WANDB_RUN_ID=unique_run_id

# Option 2: Via wandb.init() in Python
wandb.init(
    project="imagenet21k",
    resume="allow",
    id="unique_run_id"
)
```

## 🎓 Advanced Features

### 1. Custom Metrics
Add custom logging in training scripts:
```python
import wandb

# Log custom metrics
wandb.log({
    "custom/metric_name": value,
    "custom/another_metric": another_value,
})

# Log images
wandb.log({
    "examples": [wandb.Image(img) for img in sample_images]
})

# Log histograms
wandb.log({
    "gradients": wandb.Histogram(gradient_values)
})
```

### 2. Model Checkpointing with W&B
```python
# Save model as W&B artifact
artifact = wandb.Artifact('model', type='model')
artifact.add_file('model.pth')
wandb.log_artifact(artifact)

# Load model from artifact
artifact = wandb.use_artifact('user/project/model:latest')
artifact_dir = artifact.download()
model.load_state_dict(torch.load(f'{artifact_dir}/model.pth'))
```

### 3. Hyperparameter Sweeps
Create `sweep.yaml`:
```yaml
program: train_semantic_softmax_wandb.py
method: bayes
metric:
  name: val/top1_accuracy
  goal: maximize
parameters:
  lr:
    min: 1e-4
    max: 1e-3
  batch_size:
    values: [32, 64, 128]
  weight_decay:
    min: 1e-5
    max: 1e-3
```

Run sweep:
```bash
wandb sweep sweep.yaml
wandb agent sweep_id
```

### 4. Alerts
Set up alerts for:
- Training completion
- Accuracy thresholds
- Error conditions

Configure in W&B dashboard under "Alerts" tab.

## 🔍 Monitoring During Training

### Check W&B URL
```bash
# From job output
tail -f logs/train_*.out | grep "wandb.ai"

# Example output:
# Weights & Biases initialized: https://wandb.ai/username/project/runs/run_id
```

### Dashboard Sections
1. **Overview**: Summary cards with key metrics
2. **Charts**: Customizable plots and visualizations
3. **System**: Hardware utilization graphs
4. **Logs**: Console output from training
5. **Files**: Generated files and artifacts
6. **Notes**: Add markdown notes about the run

### Real-Time Monitoring Tips
- Refresh browser to see latest metrics
- Use "Live" mode for auto-refresh
- Create custom dashboards for specific metrics
- Set up mobile notifications via W&B app

## 🐛 Troubleshooting

### Issue: "wandb: ERROR Failed to connect"
```bash
# Solution 1: Check internet connection
ping wandb.ai

# Solution 2: Use offline mode
export WANDB_MODE=offline

# Solution 3: Check firewall/proxy settings
export HTTPS_PROXY=http://proxy:port
```

### Issue: "ImportError: No module named 'wandb'"
```bash
# Install wandb
pip install wandb

# Or in requirements.txt
echo "wandb" >> requirements.txt
pip install -r requirements.txt
```

### Issue: "wandb: ERROR API key not configured"
```bash
# Login interactively
wandb login

# Or set environment variable
export WANDB_API_KEY="your_key"

# Get key from: https://wandb.ai/authorize
```

### Issue: Too much disk space used
```bash
# Clean old runs
wandb sync --clean

# Limit cached runs
export WANDB_CACHE_SIZE=1GB

# Move cache to larger disk
export WANDB_CACHE_DIR=/large/disk/wandb_cache
```

### Issue: Slow training with W&B
```bash
# Reduce logging frequency
# In training script, change log interval from 100 to 500

# Disable model watching
# Comment out: wandb.watch(model)

# Disable system stats
export WANDB_DISABLE_SYSTEM_STATS=true
```

### Issue: W&B not working in distributed training
```bash
# Ensure only master process logs
# Already implemented in training scripts with is_master() check

# Verify in training output
grep "Weights & Biases initialized" logs/train_*.out
# Should appear only once
```

## 📈 Performance Impact

### Overhead
- **Negligible**: < 1% training time with default settings
- **Logging frequency**: Adjust to balance detail vs. overhead
- **Model watching**: Can add 5-10% overhead if enabled

### Recommendations
- Log every 100-500 batches (default: 100)
- Disable model watching for production runs
- Disable system stats if not needed
- Use offline mode and sync later for fastest training

## 🎯 Best Practices

### 1. Naming Convention
```bash
# Use descriptive run names
WANDB_RUN_NAME="${MODEL_NAME}_${BATCH_SIZE}bs_${LR}lr_$(date +%Y%m%d)"

# Examples:
# - tresnet_m_64bs_3e-4lr_20260114
# - resnet50_128bs_5e-4lr_experiment2
```

### 2. Project Organization
```bash
# Separate projects for different experiments
WANDB_PROJECT="imagenet21k-baseline"
WANDB_PROJECT="imagenet21k-ablation"
WANDB_PROJECT="imagenet21k-production"
```

### 3. Tag Your Runs
```python
wandb.init(
    project="imagenet21k",
    tags=["tresnet_m", "semantic_softmax", "fall11"],
)
```

### 4. Add Notes
```python
# In training script
wandb.config.update({
    "notes": "Testing new data augmentation",
    "dataset_version": "fall11",
})
```

### 5. Team Collaboration
- Use team entities: `--wandb_entity=team_name`
- Share dashboard links with teammates
- Add comments on interesting runs
- Create reports for presentation

## 📚 Additional Resources

### Official Documentation
- W&B Documentation: https://docs.wandb.ai
- PyTorch Integration: https://docs.wandb.ai/guides/integrations/pytorch
- CLI Reference: https://docs.wandb.ai/ref/cli

### Video Tutorials
- Getting Started: https://www.youtube.com/wandb
- Advanced Features: https://wandb.ai/site/tutorials

### Example Projects
- W&B Gallery: https://wandb.ai/gallery
- Community Examples: https://github.com/wandb/examples

## 🎉 Summary

**To enable W&B tracking:**

1. **Install**: `pip install wandb`
2. **Login**: `wandb login`
3. **Use W&B script**: `bsub < lsf_train_wandb.sh`
4. **View dashboard**: Check job output for W&B URL

**Benefits:**
- ✅ Real-time training visualization
- ✅ Automatic metric tracking
- ✅ Experiment comparison
- ✅ Team collaboration
- ✅ No code changes needed (use W&B-enabled scripts)

**Dashboard URL**: https://wandb.ai

Happy tracking! 🚀
