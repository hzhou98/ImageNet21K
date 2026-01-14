#!/bin/bash
# --------------------------------------------------------
# ImageNet-21K Pretraining for The Masses
# Copyright 2021 Alibaba MIIL (c)
# Licensed under MIT License [see the LICENSE file for details]
# Written by Tal Ridnik
# Modified to process OpenDataLab ImageNet-21k raw files
# --------------------------------------------------------

# Source directory with raw tar/zip files
SOURCE_DIR=/research/groups/yu3grp/projects/scRNASeq/yu3grp/hzhou98/PublicDataSets/ImageNet21K/OpenDataLab___ImageNet-21k/raw

# Target directory for extracted and processed data
ROOT=/research/groups/yu3grp/projects/scRNASeq/yu3grp/hzhou98/PublicDataSets/ImageNet21K/imagenet21k_train

# Backup directory for small classes
BACKUP=/research/groups/yu3grp/projects/scRNASeq/yu3grp/hzhou98/PublicDataSets/ImageNet21K/imagenet21k_small_classes

# Validation directory
VAL_ROOT=/research/groups/yu3grp/projects/scRNASeq/yu3grp/hzhou98/PublicDataSets/ImageNet21K/imagenet21k_val

# Create target directories
mkdir -p ${ROOT}
mkdir -p ${BACKUP}
mkdir -p ${VAL_ROOT}

echo "=========================================="
echo "ImageNet-21K Processing Script"
echo "=========================================="
echo "Source: $SOURCE_DIR"
echo "Target: $ROOT"
echo "Backup: $BACKUP"
echo "Validation: $VAL_ROOT"
echo "=========================================="

# --------------------------------------------------------
# Step 1: Unzip/Extract files from OpenDataLab raw directory
# --------------------------------------------------------

echo ""
echo "Step 1: Extracting files from raw directory..."
cd ${SOURCE_DIR}

# Count files to process
NUM_TAR=$(find . -maxdepth 1 -type f -name "*.tar" | wc -l)
NUM_TARGZ=$(find . -maxdepth 1 -type f \( -name "*.tar.gz" -o -name "*.tgz" \) | wc -l)
NUM_ZIP=$(find . -maxdepth 1 -type f -name "*.zip" | wc -l)
echo "Found archives: $NUM_TAR tar, $NUM_TARGZ tar.gz, $NUM_ZIP zip"

# Check if parallel is available
if command -v parallel &> /dev/null; then
    USE_PARALLEL=true
    echo "Using GNU parallel for faster extraction..."
else
    USE_PARALLEL=false
    echo "GNU parallel not found, using sequential extraction..."
fi

# Extract tar files
if [ $NUM_TAR -gt 0 ]; then
    echo "Extracting tar archives..."
    if [ "$USE_PARALLEL" = true ]; then
        export ROOT
        find . -maxdepth 1 -type f -name "*.tar" | parallel --will-cite 'echo "  Extracting: {}"; tar --strip-components=1 -xf {} -C "$ROOT"'
    else
        for file in *.tar; do
            if [ -f "$file" ]; then
                echo "  Extracting: $file"
                tar --strip-components=1 -xf "$file" -C ${ROOT}
            fi
        done
    fi
fi

# Extract tar.gz files
if [ $NUM_TARGZ -gt 0 ]; then
    echo "Extracting tar.gz archives..."
    if [ "$USE_PARALLEL" = true ]; then
        export ROOT
        find . -maxdepth 1 -type f \( -name "*.tar.gz" -o -name "*.tgz" \) | parallel --will-cite 'echo "  Extracting: {}"; tar --strip-components=1 -xzf {} -C "$ROOT"'
    else
        for file in *.tar.gz *.tgz; do
            if [ -f "$file" ]; then
                echo "  Extracting: $file"
                tar --strip-components=1 -xzf "$file" -C ${ROOT}
            fi
        done
    fi
fi

# Extract zip files
if [ $NUM_ZIP -gt 0 ]; then
    echo "Extracting zip archives..."
    if [ "$USE_PARALLEL" = true ]; then
        export ROOT
        find . -maxdepth 1 -type f -name "*.zip" | parallel --will-cite 'echo "  Extracting: {}"; UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -q {} -d "$ROOT"'
    else
        for file in *.zip; do
            if [ -f "$file" ]; then
                echo "  Extracting: $file"
                unzip -q "$file" -d ${ROOT}
            fi
        done
    fi
    
    # After all extractions, check and remove imagnet21k folder if it exists
    if [ -d "${ROOT}/imagnet21k" ]; then
        echo "Moving contents from imagnet21k folder..."
        mv ${ROOT}/imagnet21k/* ${ROOT}/ 2>/dev/null || true
        rmdir ${ROOT}/imagnet21k
        echo "✓ Removed intermediate imagnet21k folder"
    fi
fi

# Final check: remove any intermediate imagnet21k or imagenet21k folders after all extractions
echo ""
echo "Checking for intermediate folders..."
for folder in "${ROOT}/imagnet21k" "${ROOT}/imagenet21k"; do
    if [ -d "$folder" ]; then
        echo "Moving contents from $(basename "$folder") folder..."
        mv "$folder"/* ${ROOT}/ 2>/dev/null || true
        rmdir "$folder" 2>/dev/null || true
        echo "✓ Removed intermediate $(basename "$folder") folder"
    fi
done

echo "✓ Extraction complete!"

# --------------------------------------------------------
# Step 2: Extract individual class tar files (if any)
# --------------------------------------------------------

echo ""
echo "Step 2: Extracting individual class archives..."
cd ${ROOT}

# Check if parallel is available
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel for faster extraction..."
    find . -name "*.tar" -type f | parallel --will-cite 'echo "  Extracting: {}"; ext={/}; target_folder=${ext%.*}; mkdir -p $target_folder; tar -xf {} -C $target_folder'
else
    echo "Processing tar files sequentially..."
    find . -name "*.tar" -type f | while read tarfile; do
        echo "  Extracting: $tarfile"
        ext=$(basename "$tarfile")
        target_folder="${ext%.*}"
        mkdir -p "$target_folder"
        tar -xf "$tarfile" -C "$target_folder"
    done
fi

# Count extracted classes
NUM_CLASSES=$(find ./ -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "✓ Total extracted directories: $NUM_CLASSES"

# --------------------------------------------------------
# Step 3: Clean up tar files (SKIPPED - keeping original archives)
# --------------------------------------------------------

echo ""
echo "Step 3: Skipping cleanup of archive files..."
cd ${ROOT}
TAR_COUNT=$(find . -name "*.tar" -type f | wc -l)
echo "✓ Keeping $TAR_COUNT tar files (not removed)"

# --------------------------------------------------------
# Step 4: Remove uncommon classes for transfer learning
# --------------------------------------------------------

echo ""
echo "Step 4: Removing uncommon classes (< 500 images)..."
cd ${ROOT}

REMOVED_COUNT=0
KEPT_COUNT=0

for c in ${ROOT}/n*; do
    if [ -d "$c" ]; then
        # Count JPEG files (case-insensitive)
        count=$(find "$c" -type f \( -iname "*.jpg" -o -iname "*.jpeg" \) 2>/dev/null | wc -l)
        class_name=$(basename "$c")
        
        if [ "$count" -gt "500" ]; then
            echo "  ✓ Keep $class_name (count = $count)"
            KEPT_COUNT=$((KEPT_COUNT + 1))
        else
            echo "  ✗ Remove $class_name (count = $count)"
            mv "$c" ${BACKUP}/
            REMOVED_COUNT=$((REMOVED_COUNT + 1))
        fi
    fi
done

echo "✓ Classes kept: $KEPT_COUNT"
echo "✓ Classes removed: $REMOVED_COUNT"

# --------------------------------------------------------
# Step 5: Count valid classes
# --------------------------------------------------------

echo ""
echo "Step 5: Counting valid classes..."
cd ${ROOT}
VALID_CLASSES=$(find ./ -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "✓ Number of valid classes: $VALID_CLASSES (expected: ~11221)"

# --------------------------------------------------------
# Step 6: Create validation set
# --------------------------------------------------------

echo ""
echo "Step 6: Creating validation set (50 images per class)..."
cd ${ROOT}

VAL_CREATED=0
for i in ${ROOT}/n*; do
    if [ -d "$i" ]; then
        c=$(basename "$i")
        mkdir -p ${VAL_ROOT}/$c
        
        # Count images before moving
        total_images=$(find "$i" -type f \( -iname "*.jpg" -o -iname "*.jpeg" \) 2>/dev/null | wc -l)
        
        # Find JPEG files (case-insensitive) and move first 50 to validation
        moved=0
        find "$i" -type f \( -iname "*.jpg" -o -iname "*.jpeg" \) 2>/dev/null | head -n 50 | while read img; do
            mv "$img" ${VAL_ROOT}/$c/
            moved=$((moved + 1))
        done
        
        VAL_CREATED=$((VAL_CREATED + 1))
        if [ $((VAL_CREATED % 1000)) -eq 0 ]; then
            echo "  Processed $VAL_CREATED classes..."
        fi
    fi
done

echo "✓ Validation set created for $VAL_CREATED classes"

# --------------------------------------------------------
# Step 7: Verify dataset structure
# --------------------------------------------------------

echo ""
echo "Step 7: Verifying dataset structure..."
TRAIN_CLASSES=$(find ${ROOT} -mindepth 1 -maxdepth 1 -type d | wc -l)
VAL_CLASSES=$(find ${VAL_ROOT} -mindepth 1 -maxdepth 1 -type d | wc -l)
TRAIN_IMAGES=$(find ${ROOT} -type f \( -iname "*.jpg" -o -iname "*.jpeg" \) 2>/dev/null | wc -l)
VAL_IMAGES=$(find ${VAL_ROOT} -type f \( -iname "*.jpg" -o -iname "*.jpeg" \) 2>/dev/null | wc -l)

echo "Training classes: $TRAIN_CLASSES"
echo "Validation classes: $VAL_CLASSES"
echo "Training images: $TRAIN_IMAGES"
echo "Validation images: $VAL_IMAGES"

# Sample some classes
echo ""
echo "Sample classes:"
find ${ROOT} -mindepth 1 -maxdepth 1 -type d | head -n 5

# --------------------------------------------------------
# Step 8: Optional - Resize images
# --------------------------------------------------------

echo ""
echo "Step 8: Resizing images (optional)..."
echo "To resize images, run:"
echo "  cd $(dirname $0)"
echo "  python resize.py"
echo "  OR"
echo "  python resize_short_edge.py"

echo ""
echo "=========================================="
echo "Processing Complete!"
echo "=========================================="
echo "✓ Training data: ${ROOT}"
echo "  - Classes: $TRAIN_CLASSES"
echo "  - Images: $TRAIN_IMAGES"
echo ""
echo "✓ Validation data: ${VAL_ROOT}"
echo "  - Classes: $VAL_CLASSES"
echo "  - Images: $VAL_IMAGES"
echo ""
echo "✓ Backup (small classes): ${BACKUP}"
echo "  - Classes: $REMOVED_COUNT"
echo ""
echo "Next steps:"
echo "1. Optional: Resize images with resize.py or resize_short_edge.py"
echo "2. Update training scripts to use:"
echo "   DATA_PATH=${ROOT}"
echo "=========================================="
