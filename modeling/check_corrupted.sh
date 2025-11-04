#!/bin/bash

# Script to find corrupted .npz files
# Usage: bash check_corrupted.sh [--fix]

BASE_DIR="/Users/leeaaron/Desktop/UIUC/WatchSleepNet/DATA"
FIX_MODE=false

# Parse arguments
if [[ "$1" == "--fix" ]]; then
    FIX_MODE=true
fi

echo "================================================================================"
echo "DATASET CORRUPTION CHECKER"
echo "================================================================================"

if [ "$FIX_MODE" = true ]; then
    echo "⚠️  FIX MODE ENABLED - Corrupted files will be DELETED"
else
    echo "ℹ️  CHECK MODE - Corrupted files will be listed (not deleted)"
fi

echo "================================================================================"

CORRUPTED_FILES=()

# Function to check a single file
check_file() {
    local file="$1"

    # Try to unzip -t to test the file
    if ! python3 -c "import numpy as np; np.load('$file', allow_pickle=True)" 2>/dev/null; then
        return 1
    fi
    return 0
}

# Check SHHS_MESA_IBI
echo ""
echo "Checking SHHS_MESA_IBI..."
echo "--------------------------------------------------------------------------------"

SHHS_MESA_DIR="$BASE_DIR/SHHS_MESA_IBI"
count=0
corrupted_count=0

if [ -d "$SHHS_MESA_DIR" ]; then
    for file in "$SHHS_MESA_DIR"/*.npz; do
        if [ -f "$file" ]; then
            count=$((count + 1))

            # Show progress every 500 files
            if [ $((count % 500)) -eq 0 ]; then
                echo "  Checked $count files..."
            fi

            if ! check_file "$file"; then
                echo "❌ CORRUPTED: $(basename "$file")"
                echo "   Path: $file"
                echo "   Size: $(ls -lh "$file" | awk '{print $5}')"
                CORRUPTED_FILES+=("$file")
                corrupted_count=$((corrupted_count + 1))
            fi
        fi
    done
    echo "  Total files checked: $count"
    echo "  Corrupted files found: $corrupted_count"
else
    echo "  Directory not found: $SHHS_MESA_DIR"
fi

# Check DREAMT_PIBI_SE
echo ""
echo "Checking DREAMT_PIBI_SE..."
echo "--------------------------------------------------------------------------------"

DREAMT_DIR="$BASE_DIR/DREAMT_PIBI_SE"
count=0
corrupted_count=0

if [ -d "$DREAMT_DIR" ]; then
    for file in "$DREAMT_DIR"/*.npz; do
        if [ -f "$file" ]; then
            count=$((count + 1))

            if ! check_file "$file"; then
                echo "❌ CORRUPTED: $(basename "$file")"
                echo "   Path: $file"
                echo "   Size: $(ls -lh "$file" | awk '{print $5}')"
                CORRUPTED_FILES+=("$file")
                corrupted_count=$((corrupted_count + 1))
            fi
        fi
    done
    echo "  Total files checked: $count"
    echo "  Corrupted files found: $corrupted_count"
else
    echo "  Directory not found: $DREAMT_DIR"
fi

# Summary
echo ""
echo "================================================================================"
echo "SUMMARY"
echo "================================================================================"

if [ ${#CORRUPTED_FILES[@]} -eq 0 ]; then
    echo "✅ No corrupted files found! Your dataset is clean."
else
    echo "❌ Total corrupted files found: ${#CORRUPTED_FILES[@]}"
    echo ""
    echo "Corrupted files:"
    for file in "${CORRUPTED_FILES[@]}"; do
        echo "  $file"
    done

    if [ "$FIX_MODE" = true ]; then
        echo ""
        echo "================================================================================"
        echo "REMOVING CORRUPTED FILES"
        echo "================================================================================"

        for file in "${CORRUPTED_FILES[@]}"; do
            if rm "$file" 2>/dev/null; then
                echo "✅ Deleted: $(basename "$file")"
            else
                echo "❌ Failed to delete: $(basename "$file")"
            fi
        done

        echo ""
        echo "✅ Cleanup complete! You can now re-run training:"
        echo "   cd /Users/leeaaron/Desktop/UIUC/WatchSleepNet/modeling"
        echo "   python train_transfer.py --model=watchsleepnet"
    else
        echo ""
        echo "To remove corrupted files, run:"
        echo "  bash check_corrupted.sh --fix"
    fi
fi

echo "================================================================================"
