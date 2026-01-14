#!/bin/bash
# --------------------------------------------------------
# Pre-training Verification Script
# Checks if environment is ready for training
# --------------------------------------------------------

echo "=========================================="
echo "ImageNet-21K Training Environment Check"
echo "=========================================="
echo ""

CHECKS_PASSED=0
CHECKS_FAILED=0
WARNINGS=0

# Function to check status
check_status() {
    if [ $1 -eq 0 ]; then
        echo "✓ PASS"
        CHECKS_PASSED=$((CHECKS_PASSED + 1))
        return 0
    else
        echo "✗ FAIL"
        CHECKS_FAILED=$((CHECKS_FAILED + 1))
        return 1
    fi
}

warn_status() {
    echo "⚠ WARNING"
    WARNINGS=$((WARNINGS + 1))
}

# ========================================
# Check 1: Directory Structure
# ========================================

echo -n "1. Checking directory structure... "
if [ -d "logs" ] && [ -d "resources" ] && [ -d "pretrained_models" ]; then
    check_status 0
else
    check_status 1
    echo "   Missing directories. Run: ./setup_training.sh"
fi

# ========================================
# Check 2: Training Scripts
# ========================================

echo -n "2. Checking LSF training scripts... "
if [ -f "lsf_train_single_node.sh" ] && [ -f "lsf_train_multi_gpu.sh" ]; then
    if [ -x "lsf_train_single_node.sh" ]; then
        check_status 0
    else
        warn_status
        echo "   Scripts exist but not executable. Run: chmod +x lsf_train_*.sh"
    fi
else
    check_status 1
    echo "   Training scripts not found"
fi

# ========================================
# Check 3: Semantic Tree
# ========================================

echo -n "3. Checking semantic tree (fall11)... "
if [ -f "resources/imagenet21k_miil_tree.pth" ]; then
    SIZE=$(du -h resources/imagenet21k_miil_tree.pth | cut -f1)
    echo "✓ PASS (Size: $SIZE)"
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
else
    check_status 1
    echo "   Download with:"
    echo "   wget -O resources/imagenet21k_miil_tree.pth \\"
    echo "     https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/resources/fall11/imagenet21k_miil_tree.pth"
fi

# ========================================
# Check 4: Pretrained Models
# ========================================

echo -n "4. Checking pretrained models... "
PRETRAINED_COUNT=$(find pretrained_models -name "*.pth" 2>/dev/null | wc -l)
if [ $PRETRAINED_COUNT -gt 0 ]; then
    echo "✓ PASS ($PRETRAINED_COUNT models found)"
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
    echo "   Available models:"
    ls -lh pretrained_models/*.pth 2>/dev/null | awk '{print "   - " $9 " (" $5 ")"}'
else
    warn_status
    echo "   No pretrained models found (optional but recommended)"
    echo "   Run: ./setup_training.sh to download"
fi

# ========================================
# Check 5: Python Environment
# ========================================

echo -n "5. Checking Python version... "
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    echo "✓ PASS (Python $PYTHON_VERSION)"
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
else
    check_status 1
    echo "   Python3 not found"
fi

# ========================================
# Check 6: PyTorch
# ========================================

echo -n "6. Checking PyTorch installation... "
python3 -c "import torch; print('✓ PASS (PyTorch ' + torch.__version__ + ')')" 2>/dev/null
if [ $? -eq 0 ]; then
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
else
    check_status 1
    echo "   Install with: pip install torch torchvision"
fi

# ========================================
# Check 7: CUDA Availability
# ========================================

echo -n "7. Checking CUDA availability (Python)... "
python3 -c "import torch; assert torch.cuda.is_available(); print('✓ PASS (' + str(torch.cuda.device_count()) + ' GPUs)')" 2>/dev/null
if [ $? -eq 0 ]; then
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
else
    warn_status
    echo "   No CUDA GPUs detected (normal on login nodes)"
fi

# ========================================
# Check 8: Required Python Packages
# ========================================

echo "8. Checking required Python packages..."
REQUIRED_PACKAGES=("torch" "torchvision" "timm" "PIL" "numpy")
PACKAGES_OK=true

for pkg in "${REQUIRED_PACKAGES[@]}"; do
    echo -n "   - $pkg... "
    python3 -c "import $pkg" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "✓"
    else
        echo "✗"
        PACKAGES_OK=false
    fi
done

if [ "$PACKAGES_OK" = true ]; then
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
else
    CHECKS_FAILED=$((CHECKS_FAILED + 1))
    echo "   Install missing packages: pip install -r requirements.txt"
fi

# ========================================
# Check 9: Data Path
# ========================================

echo -n "9. Checking ImageNet-21K data path... "
DATA_PATH="/research/groups/yu3grp/projects/scRNASeq/yu3grp/hzhou98/PublicDataSets/ImageNet21K/fall11"

if [ -d "$DATA_PATH" ]; then
    NUM_CLASSES=$(find $DATA_PATH -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    echo "✓ PASS ($NUM_CLASSES classes found)"
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
    
    if [ $NUM_CLASSES -lt 10000 ]; then
        warn_status
        echo "   Expected ~11221 classes, found $NUM_CLASSES"
        echo "   Verify dataset is fully processed"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    check_status 1
    echo "   Data path not found: $DATA_PATH"
    echo "   Process dataset with: cd dataset_preprocessing && bash processing_script_new.sh"
fi

# ========================================
# Check 10: LSF Environment
# ========================================

echo -n "10. Checking LSF availability... "
if command -v bsub &> /dev/null; then
    echo "✓ PASS"
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
else
    warn_status
    echo "    bsub command not found (normal on non-LSF systems)"
fi

# ========================================
# Check 11: Disk Space
# ========================================

echo -n "11. Checking disk space... "
AVAILABLE_SPACE=$(df -h . | awk 'NR==2 {print $4}')
AVAILABLE_GB=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')

echo "(Available: $AVAILABLE_SPACE)"
if [ $AVAILABLE_GB -gt 100 ]; then
    echo "    ✓ Sufficient space for training outputs"
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
else
    warn_status
    echo "    Low disk space. Recommended: >100GB for outputs"
fi

# ========================================
# Check 12: Documentation
# ========================================

echo -n "12. Checking documentation... "
if [ -f "LSF_TRAINING_GUIDE.md" ] && [ -f "QUICK_START.md" ]; then
    echo "✓ PASS"
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
else
    warn_status
    echo "    Documentation files missing"
fi

# ========================================
# Summary
# ========================================

echo ""
echo "=========================================="
echo "Verification Summary"
echo "=========================================="
echo "Checks passed:  $CHECKS_PASSED"
echo "Checks failed:  $CHECKS_FAILED"
echo "Warnings:       $WARNINGS"
echo ""

if [ $CHECKS_FAILED -eq 0 ]; then
    echo "✓ System is READY for training!"
    echo ""
    echo "Next steps:"
    echo "1. Review and edit a training script:"
    echo "   nano lsf_train_single_node.sh"
    echo ""
    echo "2. Submit your job:"
    echo "   bsub < lsf_train_single_node.sh"
    echo "   OR"
    echo "   ./submit_job.sh (interactive)"
    echo ""
    echo "3. Monitor training:"
    echo "   bjobs"
    echo "   tail -f logs/train_*.out"
    echo ""
    EXIT_CODE=0
else
    echo "✗ System has $CHECKS_FAILED FAILED checks"
    echo ""
    echo "Please resolve the failed checks before training."
    echo "Run ./setup_training.sh to fix common issues."
    echo ""
    EXIT_CODE=1
fi

if [ $WARNINGS -gt 0 ]; then
    echo "⚠ There are $WARNINGS warnings (non-critical)"
    echo ""
fi

echo "For detailed setup instructions, see:"
echo "  - QUICK_START.md (quick guide)"
echo "  - LSF_TRAINING_GUIDE.md (comprehensive)"
echo "  - README_LSF.md (scripts overview)"
echo ""

exit $EXIT_CODE
