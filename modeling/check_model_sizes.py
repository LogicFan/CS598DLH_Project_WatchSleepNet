"""
Script to calculate model sizes and estimate storage requirements.

This script instantiates each model and calculates:
1. Total parameters
2. Trainable parameters
3. Estimated checkpoint size
4. Total storage requirements for training
"""

import torch
import os
from pathlib import Path

from config import (
    WatchSleepNetConfig,
    InsightSleepNetConfig,
    SleepConvNetConfig,
    SleepPPGNetConfig,
)
from engine import setup_model_and_optimizer

DEVICE = "cpu"  # Use CPU for parameter counting

def count_parameters(model):
    """Count total and trainable parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def estimate_checkpoint_size(total_params):
    """
    Estimate checkpoint file size in MB.

    Each parameter is stored as a 32-bit float (4 bytes).
    Plus overhead for optimizer state (roughly 2x for Adam).
    """
    # Model parameters (4 bytes per parameter)
    model_size_mb = (total_params * 4) / (1024 ** 2)

    # Add optimizer state (Adam uses ~2x model parameters)
    # Training checkpoint = model + optimizer state
    training_checkpoint_mb = model_size_mb * 3

    # Inference checkpoint = model only
    inference_checkpoint_mb = model_size_mb

    return inference_checkpoint_mb, training_checkpoint_mb

def get_dataset_sizes():
    """Get the sizes of preprocessed datasets."""
    base_dir = Path("/Users/leeaaron/Desktop/UIUC/WatchSleepNet/DATA")

    datasets = {
        "SHHS_MESA_IBI": base_dir / "SHHS_MESA_IBI",
        "DREAMT_PIBI_SE": base_dir / "DREAMT_PIBI_SE",
    }

    sizes = {}
    for name, path in datasets.items():
        if path.exists():
            # Calculate total size
            total_size = 0
            for file in path.glob("*.npz"):
                total_size += os.path.getsize(file)
            sizes[name] = total_size / (1024 ** 3)  # Convert to GB
        else:
            sizes[name] = 0

    return sizes

# Model configurations
MODEL_CONFIGS = {
    "WatchSleepNet": WatchSleepNetConfig,
    "InsightSleepNet": InsightSleepNetConfig,
    "SleepConvNet": SleepConvNetConfig,
    "SleepPPGNet": SleepPPGNetConfig,
}

print("="*80)
print("MODEL SIZE ANALYSIS")
print("="*80)
print()

results = {}

for model_name, config_class in MODEL_CONFIGS.items():
    print(f"{'='*80}")
    print(f"Model: {model_name}")
    print(f"{'='*80}")

    # Get configuration
    config_dict = config_class.to_dict()

    # Instantiate model
    try:
        model, _ = setup_model_and_optimizer(
            model_name=model_name.lower().replace(" ", ""),
            model_params=config_dict,
            device=DEVICE,
            saved_model_path=None,
            learning_rate=config_dict["LEARNING_RATE"],
            weight_decay=config_dict["WEIGHT_DECAY"],
            freeze_layers=False
        )

        # Count parameters
        total_params, trainable_params = count_parameters(model)

        # Estimate sizes
        inference_size, training_size = estimate_checkpoint_size(total_params)

        # Store results
        results[model_name] = {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "inference_size_mb": inference_size,
            "training_size_mb": training_size,
        }

        print(f"Total Parameters:      {total_params:,}")
        print(f"Trainable Parameters:  {trainable_params:,}")
        print(f"Inference Checkpoint:  ~{inference_size:.2f} MB")
        print(f"Training Checkpoint:   ~{training_size:.2f} MB")
        print()

    except Exception as e:
        print(f"Error instantiating {model_name}: {e}")
        print()

print("="*80)
print("STORAGE REQUIREMENTS SUMMARY")
print("="*80)
print()

# Dataset sizes
print("Dataset Storage:")
dataset_sizes = get_dataset_sizes()
total_dataset_gb = 0
for name, size_gb in dataset_sizes.items():
    print(f"  {name:20s}: {size_gb:6.2f} GB")
    total_dataset_gb += size_gb
print(f"  {'Total Datasets':20s}: {total_dataset_gb:6.2f} GB")
print()

# Checkpoint storage per model
print("Checkpoint Storage (per model):")
print("  Type                   WatchSleepNet  InsightSleepNet  SleepConvNet  SleepPPGNet")
print("  " + "-"*75)

# Pretraining checkpoint (1 per model)
print("  Pretraining (1x):      ", end="")
for model in ["WatchSleepNet", "InsightSleepNet", "SleepConvNet", "SleepPPGNet"]:
    if model in results:
        print(f"{results[model]['inference_size_mb']:7.1f} MB    ", end="")
print()

# Fine-tuning checkpoints (5-fold CV, so 5 per model)
print("  Fine-tuning (5x):      ", end="")
for model in ["WatchSleepNet", "InsightSleepNet", "SleepConvNet", "SleepPPGNet"]:
    if model in results:
        total_finetuning = results[model]['inference_size_mb'] * 5
        print(f"{total_finetuning:7.1f} MB    ", end="")
print()

# Total per model
print("  Total per model:       ", end="")
for model in ["WatchSleepNet", "InsightSleepNet", "SleepConvNet", "SleepPPGNet"]:
    if model in results:
        total = results[model]['inference_size_mb'] * 6  # 1 pretrain + 5 finetune
        print(f"{total:7.1f} MB    ", end="")
print()
print()

# Overall storage estimates
print("Overall Storage Estimates:")
print(f"  Datasets:                {total_dataset_gb:.2f} GB")

# Calculate total checkpoint storage for all models
total_checkpoints_mb = 0
for model, data in results.items():
    # 1 pretraining + 5 fine-tuning checkpoints per model
    total_checkpoints_mb += data['inference_size_mb'] * 6

print(f"  All Model Checkpoints:   {total_checkpoints_mb / 1024:.2f} GB")
print(f"  Training Logs/Outputs:   ~0.10 GB (estimated)")
print()

total_storage = total_dataset_gb + (total_checkpoints_mb / 1024) + 0.10
print(f"  TOTAL ESTIMATED:         {total_storage:.2f} GB")
print()

print("="*80)
print("NOTES")
print("="*80)
print("- Checkpoint sizes are for inference (model weights only)")
print("- During training, checkpoints temporarily include optimizer state (~3x larger)")
print("- Only best checkpoints are kept after training (inference size)")
print("- Each model gets 1 pretraining checkpoint + 5 fine-tuning checkpoints")
print("- Training logs and intermediate outputs add minimal storage (~100 MB)")
print()
