"""
Script to check for corrupted .npz files in datasets.

This script scans through all .npz files and identifies any that are corrupted
(cannot be loaded). Corrupted files will be listed for removal.
"""

import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

def check_file(file_path):
    """
    Check if a .npz file can be loaded.

    Returns:
        (bool, str): (is_valid, error_message)
    """
    try:
        data = np.load(file_path, allow_pickle=True)
        # Try to access the data to ensure it's really valid
        _ = data.files
        data.close()
        return True, None
    except Exception as e:
        return False, str(e)

def scan_dataset(dataset_path, fix=False):
    """
    Scan a dataset directory for corrupted .npz files.

    Args:
        dataset_path: Path to the dataset directory
        fix: If True, delete corrupted files

    Returns:
        List of corrupted file paths
    """
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        print(f"Dataset path does not exist: {dataset_path}")
        return []

    # Get all .npz files
    npz_files = list(dataset_path.glob("*.npz"))

    if not npz_files:
        print(f"No .npz files found in {dataset_path}")
        return []

    print(f"\nScanning {len(npz_files)} files in {dataset_path.name}...")

    corrupted_files = []

    for file_path in tqdm(npz_files, desc="Checking files"):
        is_valid, error_msg = check_file(file_path)

        if not is_valid:
            corrupted_files.append((file_path, error_msg))
            print(f"\n❌ CORRUPTED: {file_path.name}")
            print(f"   Error: {error_msg}")

    if corrupted_files:
        print(f"\n{'='*80}")
        print(f"FOUND {len(corrupted_files)} CORRUPTED FILE(S)")
        print(f"{'='*80}")

        for file_path, error_msg in corrupted_files:
            print(f"\n{file_path}")
            print(f"  Size: {os.path.getsize(file_path)} bytes")
            print(f"  Error: {error_msg}")

        if fix:
            print(f"\n{'='*80}")
            print("REMOVING CORRUPTED FILES")
            print(f"{'='*80}")

            for file_path, _ in corrupted_files:
                try:
                    os.remove(file_path)
                    print(f"✅ Deleted: {file_path.name}")
                except Exception as e:
                    print(f"❌ Failed to delete {file_path.name}: {e}")
        else:
            print(f"\n{'='*80}")
            print("TO REMOVE CORRUPTED FILES, RUN:")
            print(f"{'='*80}")
            print(f"python check_corrupted_files.py --fix")
    else:
        print(f"\n✅ All {len(npz_files)} files are valid!")

    return [f[0] for f in corrupted_files]

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Check for corrupted .npz files in datasets"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Automatically delete corrupted files"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "shhs_mesa_ibi", "dreamt_pibi_se"],
        help="Which dataset to check (default: all)"
    )

    args = parser.parse_args()

    base_dir = Path("/Users/leeaaron/Desktop/UIUC/WatchSleepNet/DATA")

    datasets = {
        "shhs_mesa_ibi": base_dir / "SHHS_MESA_IBI",
        "dreamt_pibi_se": base_dir / "DREAMT_PIBI_SE",
    }

    print("="*80)
    print("DATASET CORRUPTION CHECKER")
    print("="*80)

    if args.fix:
        print("⚠️  FIX MODE ENABLED - Corrupted files will be DELETED")
    else:
        print("ℹ️  CHECK MODE - Corrupted files will be listed (not deleted)")

    print("="*80)

    total_corrupted = []

    if args.dataset == "all":
        # Check all datasets
        for name, path in datasets.items():
            corrupted = scan_dataset(path, fix=args.fix)
            total_corrupted.extend(corrupted)
    else:
        # Check specific dataset
        path = datasets.get(args.dataset)
        if path:
            corrupted = scan_dataset(path, fix=args.fix)
            total_corrupted.extend(corrupted)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if total_corrupted:
        print(f"❌ Total corrupted files found: {len(total_corrupted)}")
        if args.fix:
            print(f"✅ Corrupted files have been removed")
            print(f"\nYou can now re-run training:")
            print(f"  python train_transfer.py --model=watchsleepnet")
        else:
            print(f"\nTo remove corrupted files, run:")
            print(f"  python check_corrupted_files.py --fix")
    else:
        print("✅ No corrupted files found! Your dataset is clean.")

    print("="*80)
