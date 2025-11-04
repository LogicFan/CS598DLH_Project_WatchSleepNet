"""
Standalone evaluation script for trained WatchSleepNet models.

This script loads a trained model checkpoint and evaluates it on a specified dataset.
It computes comprehensive metrics including accuracy, F1 score, Cohen's Kappa,
REM F1 score, AUROC, and per-AHI category metrics.

Usage Examples:
    # Evaluate WatchSleepNet on DREAMT dataset
    python evaluate_model.py --model watchsleepnet --dataset dreamt_pibi --checkpoint checkpoints/watchsleepnet/dreamt_pibi/best_saved_model_ablation_separate_pretraining_fold1.pt

    # Evaluate InsightSleepNet on SHHS+MESA dataset
    python evaluate_model.py --model insightsleepnet --dataset shhs_mesa_ibi --checkpoint checkpoints/insightsleepnet/shhs_mesa_ibi/best_saved_model_ablation_separate_pretraining.pt

    # Evaluate with AHI category breakdown
    python evaluate_model.py --model watchsleepnet --dataset dreamt_pibi --checkpoint checkpoints/watchsleepnet/dreamt_pibi/best_saved_model_ablation_separate_pretraining_fold1.pt --ahi_breakdown

    # Adjust batch size and workers
    python evaluate_model.py --model watchsleepnet --dataset dreamt_pibi --checkpoint checkpoints/watchsleepnet/dreamt_pibi/best_saved_model_ablation_separate_pretraining_fold1.pt --batch_size 8 --num_workers 4
"""

import os
import torch
import argparse
import numpy as np
from pathlib import Path

from config import (
    dataset_configurations,
    WatchSleepNetConfig,
    InsightSleepNetConfig,
    SleepConvNetConfig,
    SleepPPGNetConfig,
)
from engine import setup_model_and_optimizer, test_step, compute_metrics, compute_metrics_per_ahi_category
from data_setup import SSDataset
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model configuration mapping
MODEL_CONFIGS = {
    "watchsleepnet": WatchSleepNetConfig,
    "insightsleepnet": InsightSleepNetConfig,
    "sleepconvnet": SleepConvNetConfig,
    "sleepppgnet": SleepPPGNetConfig,
}


def evaluate_model_on_dataset(
    model_name,
    dataset_name,
    checkpoint_path,
    batch_size=16,
    num_workers=8,
    ahi_breakdown=False,
    task="sleep_staging"
):
    """
    Load a trained model and evaluate it on a specified dataset.

    Args:
        model_name: Name of the model architecture
        dataset_name: Name of the dataset to evaluate on
        checkpoint_path: Path to the model checkpoint (.pt file)
        batch_size: Batch size for evaluation
        num_workers: Number of data loading workers
        ahi_breakdown: Whether to compute per-AHI category metrics
        task: Task type ("sleep_staging" or "sleep_wake")

    Returns:
        Dictionary containing evaluation metrics
    """

    # Get dataset configuration
    dataset_config = dataset_configurations.get(dataset_name, None)
    if dataset_config is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Get model configuration
    model_config_class = MODEL_CONFIGS.get(model_name, None)
    if model_config_class is None:
        raise ValueError(f"Unknown model: {model_name}")

    model_config_dict = model_config_class.to_dict()

    print("="*80)
    print(f"EVALUATION: {model_name.upper()} on {dataset_name.upper()}")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {dataset_config['directory']}")
    print(f"Device: {DEVICE}")
    print(f"Task: {task}")
    print()

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Create dataset
    print("Loading dataset...")
    dataset = SSDataset(
        dir=dataset_config["directory"],
        dataset=dataset_name,
        multiplier=dataset_config["multiplier"],
        downsample_rate=dataset_config["downsampling_rate"],
        task=task,
        return_file_name=False
    )
    print(f"Dataset size: {len(dataset)} subjects")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=SSDataset.collate_fn
    )

    # Load model
    print("\nLoading model...")
    model, _ = setup_model_and_optimizer(
        model_name=model_name,
        model_params=model_config_dict,
        device=DEVICE,
        saved_model_path=None,
        learning_rate=model_config_dict["LEARNING_RATE"],
        weight_decay=model_config_dict["WEIGHT_DECAY"],
        freeze_layers=False
    )

    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Run evaluation
    print("\nRunning evaluation...")
    true_labels, predicted_labels, predicted_probs, ahi_values = test_step(
        model, dataloader, DEVICE, task=task
    )

    # Compute overall metrics
    print("\n" + "="*80)
    print("OVERALL METRICS")
    print("="*80)
    acc, f1, kappa, rem_f1, auroc = compute_metrics(
        predicted_labels,
        true_labels,
        pred_probs=predicted_probs,
        testing=True,
        task=task,
        print_conf_matrix=True,
        category_name="Overall"
    )

    print(f"\nAccuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print(f"REM F1 Score: {rem_f1:.4f}")
    print(f"AUROC: {auroc:.4f}")

    results = {
        "accuracy": acc,
        "f1_score": f1,
        "cohens_kappa": kappa,
        "rem_f1_score": rem_f1,
        "auroc": auroc
    }

    # Compute per-AHI category metrics if requested
    if ahi_breakdown:
        print("\n" + "="*80)
        print("PER-AHI CATEGORY METRICS")
        print("="*80)
        category_metrics = compute_metrics_per_ahi_category(
            true_labels, predicted_labels, predicted_probs, ahi_values
        )

        for category, metrics in category_metrics.items():
            print(f"\n{category} AHI:")
            for metric_name, metric_value in metrics.items():
                print(f"  {metric_name}: {metric_value:.4f}")

        results["ahi_category_metrics"] = category_metrics

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on a specified dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python evaluate_model.py --model watchsleepnet --dataset dreamt_pibi --checkpoint checkpoints/watchsleepnet/dreamt_pibi/best_saved_model_ablation_separate_pretraining_fold1.pt

  # With AHI breakdown
  python evaluate_model.py --model watchsleepnet --dataset dreamt_pibi --checkpoint checkpoints/watchsleepnet/dreamt_pibi/best_saved_model_ablation_separate_pretraining_fold1.pt --ahi_breakdown

  # Custom batch size
  python evaluate_model.py --model watchsleepnet --dataset dreamt_pibi --checkpoint checkpoints/watchsleepnet/dreamt_pibi/best_saved_model_ablation_separate_pretraining_fold1.pt --batch_size 8 --num_workers 4
        """
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["watchsleepnet", "insightsleepnet", "sleepconvnet", "sleepppgnet"],
        help="Model architecture to evaluate"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["dreamt_pibi", "shhs_mesa_ibi", "mesa_pibi", "shhs_ibi", "dreamt_ppg", "mesa_ppg"],
        help="Dataset to evaluate on"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for evaluation (default: 16)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of data loading workers (default: 8)"
    )
    parser.add_argument(
        "--ahi_breakdown",
        action="store_true",
        help="Compute per-AHI category metrics (Normal, Mild, Moderate, Severe)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="sleep_staging",
        choices=["sleep_staging", "sleep_wake"],
        help="Task type (default: sleep_staging)"
    )

    args = parser.parse_args()

    # Run evaluation
    results = evaluate_model_on_dataset(
        model_name=args.model,
        dataset_name=args.dataset,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        ahi_breakdown=args.ahi_breakdown,
        task=args.task
    )

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
