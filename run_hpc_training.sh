#!/bin/bash

# HPC Training Pipeline Script
# Run this script from the 'Core' directory: ./run_hpc_training.sh

echo "========================================================"
echo "   Starting End-to-End Training Pipeline on HPC"
echo "========================================================"

# 1. Dataset Generation
echo "[Step 1] Generating Combined Metadata (NIH + OpenI + ReX)..."
python3 create_full_dataset.py

if [ $? -ne 0 ]; then
    echo "Error: Dataset generation failed!"
    exit 1
fi

echo "Dataset metadata created successfully."

# 2. Training
# Parameters optimized for High AUROC:
# - Epochs: 30 (Sufficient for convergence without massive overfitting)
# - Batch Size: 32 (Stability)
# - Learning Rate: 5e-5 (Lower for fine-tuning ConvNeXt)
echo "[Step 2] Starting Training with ConvNeXt Large..."
python3 train_v3.py --epochs 30 --batch_size 32 --lr 5e-5

if [ $? -ne 0 ]; then
    echo "Error: Training failed!"
    exit 1
fi

echo "========================================================"
echo "   Training Complete!"
echo "   Best Model Saved to: models/best_model_v3.pth"
echo "========================================================"
