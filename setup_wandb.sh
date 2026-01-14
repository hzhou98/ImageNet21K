#!/bin/bash
# Quick setup script for Weights & Biases integration

echo "=========================================="
echo "Weights & Biases Setup for ImageNet-21K"
echo "=========================================="
echo ""

# Check if wandb is installed
echo -n "Checking if wandb is installed... "
python3 -c "import wandb" 2>/dev/null
if [ $? -eq 0 ]; then
    WANDB_VERSION=$(python3 -c "import wandb; print(wandb.__version__)")
    echo "✓ wandb $WANDB_VERSION is installed"
else
    echo "✗ wandb not installed"
    echo ""
    read -p "Install wandb now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Installing wandb..."
        pip install wandb
        if [ $? -eq 0 ]; then
            echo "✓ wandb installed successfully"
        else
            echo "✗ Failed to install wandb"
            exit 1
        fi
    else
        echo "Please install wandb manually: pip install wandb"
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "W&B Login Configuration"
echo "=========================================="
echo ""

# Check if already logged in
python3 -c "import wandb; wandb.login(relogin=True, key='')" 2>&1 | grep -q "Logged in" && LOGGED_IN=true || LOGGED_IN=false

if [ "$LOGGED_IN" = true ]; then
    echo "✓ You are already logged in to W&B"
    echo ""
    read -p "Do you want to login again? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping login"
        SKIP_LOGIN=true
    fi
fi

if [ "$SKIP_LOGIN" != true ]; then
    echo "To login to Weights & Biases, you need an API key."
    echo ""
    echo "Choose login method:"
    echo "1. Interactive login (opens browser)"
    echo "2. Manual API key entry"
    echo "3. Skip (login later manually)"
    echo ""
    read -p "Choice (1-3): " choice

    case $choice in
        1)
            echo "Launching interactive login..."
            wandb login
            ;;
        2)
            echo ""
            echo "Get your API key from: https://wandb.ai/authorize"
            echo ""
            read -p "Enter your W&B API key: " api_key
            if [ ! -z "$api_key" ]; then
                wandb login $api_key
                if [ $? -eq 0 ]; then
                    echo "✓ Successfully logged in"
                else
                    echo "✗ Login failed"
                fi
            fi
            ;;
        3)
            echo "Skipping login. You can login later with: wandb login"
            ;;
        *)
            echo "Invalid choice"
            ;;
    esac
fi

echo ""
echo "=========================================="
echo "W&B Configuration"
echo "=========================================="
echo ""

# Get username/entity
echo "What is your W&B username or team name?"
echo "(This will be used in LSF scripts)"
echo "Leave empty to use default"
read -p "W&B Entity: " wandb_entity

# Get default project name
echo ""
echo "Default project name for ImageNet-21K experiments?"
read -p "Project name [default: imagenet21k-training]: " wandb_project
if [ -z "$wandb_project" ]; then
    wandb_project="imagenet21k-training"
fi

echo ""
echo "=========================================="
echo "Update LSF Scripts"
echo "=========================================="
echo ""

# Update lsf_train_wandb.sh with user's settings
if [ -f "lsf_train_wandb.sh" ]; then
    if [ ! -z "$wandb_entity" ]; then
        echo "Updating WANDB_ENTITY in lsf_train_wandb.sh..."
        # Use perl for cross-platform sed compatibility
        perl -pi -e "s/WANDB_ENTITY=\"\"/WANDB_ENTITY=\"$wandb_entity\"/g" lsf_train_wandb.sh
    fi
    
    echo "Updating WANDB_PROJECT in lsf_train_wandb.sh..."
    perl -pi -e "s/WANDB_PROJECT=\"imagenet21k-training\"/WANDB_PROJECT=\"$wandb_project\"/g" lsf_train_wandb.sh
    
    echo "✓ LSF script updated"
else
    echo "⚠ lsf_train_wandb.sh not found"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Test W&B locally (optional):"
echo "   python train_semantic_softmax_wandb.py \\"
echo "       --data_path=/path/to/data \\"
echo "       --model_name=tresnet_m \\"
echo "       --epochs=1 \\"
echo "       --use_wandb \\"
echo "       --wandb_project=$wandb_project"
echo ""
echo "2. Submit LSF job with W&B:"
echo "   bsub < lsf_train_wandb.sh"
echo ""
echo "3. Monitor your training:"
echo "   https://wandb.ai/$wandb_entity/$wandb_project"
echo ""
echo "For detailed instructions, see WANDB_GUIDE.md"
echo ""

# Create a .wandb_config file for reference
cat > .wandb_config << EOF
# W&B Configuration
# Generated: $(date)

WANDB_ENTITY="$wandb_entity"
WANDB_PROJECT="$wandb_project"

# To use these in your environment:
# source .wandb_config
EOF

echo "Configuration saved to: .wandb_config"
echo ""
