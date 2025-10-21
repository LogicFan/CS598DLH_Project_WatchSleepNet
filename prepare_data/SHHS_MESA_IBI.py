import numpy as np
import biosppy
import neurokit2 as nk
import logging
from multiprocessing import Pool, cpu_count
import pandas as pd
from scipy.signal import resample_poly
from pathlib import Path

logging.basicConfig(level=logging.INFO)


def calculate_ibi_ecg(signal, fs):
    """
    Faithfully reproduces the old "biosppy" ECG approach:
      1) Run biosppy ECG pipeline.
      2) Compute IBI between consecutive R-peaks.
      3) Assign those IBIs to each sample between peaks (float64).
      4) Zero out IBIs >= 2.0 seconds.
    """
    out = biosppy.signals.ecg.ecg(signal, sampling_rate=fs, show=False)
    r_peaks = out["rpeaks"]  # indices of R-peaks
    ibi_values = np.diff(r_peaks) / fs  # IBI in seconds

    # Create an array to hold IBI (float64)
    ibi = np.zeros(signal.shape, dtype=np.float64)

    # Fill in IBI values between consecutive peaks
    for i in range(len(r_peaks) - 1):
        ibi[r_peaks[i] : r_peaks[i + 1]] = ibi_values[i]

    # Zero out improbable IBIs
    ibi[ibi >= 2.0] = 0.0
    return ibi


def calculate_ibi_ppg(signal, fs):
    """
    Faithfully reproduces the old "neurokit" PPG approach:
      1) Use neurokit2 PPG pipeline.
      2) Compute IBI between consecutive PPG peaks.
      3) Assign those IBIs to each sample between peaks (float64).
      4) Zero out IBIs >= 2.0 seconds.
    """
    signals, info = nk.ppg_process(signal, sampling_rate=fs)
    peaks = info["PPG_Peaks"]  # indices of PPG peaks
    ibi_values = np.diff(peaks) / fs

    # Create an array to hold IBI (float64)
    ibi = np.zeros(signal.shape, dtype=np.float64)

    # Fill in IBI values between consecutive peaks
    for i in range(len(peaks) - 1):
        ibi[peaks[i] : peaks[i + 1]] = ibi_values[i]

    # Zero out improbable IBIs
    ibi[ibi >= 2.0] = 0.0
    return ibi


def process_file(
    filename,
    in_dir,
    out_dir,
    info_df,
    id_col,
    ahi_col,
    fs_col,
    method,
    dataset_name
):
    """
    Loads .npz file from 'in_dir', computes continuous IBI array using the
    EXACT old code approach:
      - 'biosppy' for ECG (SHHS)
      - 'neurokit' for PPG (MESA)
    Then it downsamples from fs to 25 Hz via a smarter approach:
      - integer slicing if original_fs is a multiple of 25
      - polyphase resampling (scipy.signal.resample_poly) otherwise.
    Saves the new NPZ file (IBI float64, fs=25, downsampled stages, AHI).
    """
    file_path = in_dir / filename

    try:
        data = np.load(file_path)
        signal = data["data"].flatten()  # original waveform
        original_fs = int(data[fs_col].item())  # e.g., 125, 250, 256, etc.
        stages = data["stages"]

        # Compute IBI using the old code approach
        if method == "biosppy":  # ECG for SHHS
            ibi = calculate_ibi_ecg(signal, original_fs)
        elif method == "neurokit":  # PPG for MESA
            ibi = calculate_ibi_ppg(signal, original_fs)
        else:
            raise ValueError(f"Unknown method: {method}")

        if ibi is None:
            return

        # Parse subject ID from filename
        if dataset_name == "SHHS":
            file_id = filename.split("-")[1].split(".npz")[0].lstrip("0")
        else:  # MESA or other
            if filename.startswith("mesa-"):
                # Remove the prefix "mesa-" and the suffix ".npz"
                file_id = filename[len("mesa-") : -len(".npz")]
            else:
                file_id = filename.split(".npz")[0]

        # Retrieve AHI from info_df
        matching_rows = info_df[info_df[id_col] == file_id.lstrip("0")]
        if matching_rows.empty:
            logging.warning(f"AHI not found for {filename} in dataset {dataset_name}.")
            return
        ahi = float(matching_rows[ahi_col].values[0])

        # Smarter downsampling to 25 Hz
        target_fs = 25
        if original_fs % target_fs == 0:
            # When original_fs is an integer multiple of 25, simple slicing works
            factor = int(original_fs // target_fs)
            ibi_ds = ibi[::factor]
            stages_ds = stages[::factor]
        else:
            # For non-integer resampling factors, use polyphase resampling
            ibi_ds = resample_poly(ibi, target_fs, original_fs)
            stages_ds = resample_poly(stages, target_fs, original_fs)

        # Construct the output filename
        if dataset_name == "MESA":
            out_filename = f"mesa-{file_id}.npz"
        else:
            out_filename = filename

        # Save the processed file
        out_path = out_dir / out_filename
        np.savez(
            out_path,
            data=ibi_ds.astype(np.float64),
            fs=target_fs,
            stages=stages_ds,
            ahi=ahi
        )
        logging.info(f"{dataset_name} processed & saved => {out_filename}")

    except Exception as e:
        logging.exception(f"Error processing file: {filename} in {dataset_name}")


def process_dataset(
    dataset_name,
    in_dir,
    out_dir,
    info_path,
    id_col,
    ahi_col,
    fs_col,
    method
):
    """
    Main function:
      - Load dataset's CSV (for AHI, etc.).
      - Create out_dir.
      - Process each .npz file in parallel.
    """
    try:
        logging.info(f"=== Processing {dataset_name} dataset ===")

        in_dir = Path(in_dir)
        out_dir = Path(out_dir)
        info_path = Path(info_path)

        info_df = pd.read_csv(info_path, dtype={id_col: str})
        info_df[id_col] = info_df[id_col].str.lstrip("0")

        out_dir.mkdir(parents=True, exist_ok=True)

        file_list = sorted([f.name for f in in_dir.glob("*.npz")])
        logging.info(f"{dataset_name}: Found {len(file_list)} .npz files in {in_dir}")

        # Decide how many CPU cores to use
        num_processes = max(1, cpu_count() // 2)
        logging.info(f"{dataset_name}: Using {num_processes} processes")

        # Prepare arguments
        args_list = [
            (
                f,
                in_dir,
                out_dir,
                info_df,
                id_col,
                ahi_col,
                fs_col,
                method,
                dataset_name
            )
            for f in file_list
        ]

        # Process in parallel
        with Pool(num_processes) as pool:
            pool.starmap(process_file, args_list)

        logging.info(f"=== Completed processing {dataset_name} ===\n")

    except Exception as e:
        logging.exception(f"Error processing dataset: {dataset_name}")


if __name__ == "__main__":
    """
    This script processes two datasets in one go:
      1) SHHS (ECG) via biosppy
      2) MESA (PPG) via neurokit

    Both resulting NPZ files (one for each record) will be placed in
    the same output directory, data/processed/SHHS_MESA_IBI/.
    """
    logging.info("Starting dataset processing...")
    ROOT_PATH = Path(__file__).resolve().parent.parent
    OUTPUT_PATH = ROOT_PATH / "data/processed/SHHS_MESA_IBI"

    INPUT_SHHS_PATH = ROOT_PATH / "data/processed/SHHS_ECG"
    INPUT_SHHS_INFO = ROOT_PATH / "data/raw/SHHS/datasets/shhs-harmonized-dataset-0.21.0.csv"

    process_dataset(
        dataset_name="SHHS",
        in_dir=INPUT_SHHS_PATH,
        out_dir=OUTPUT_PATH,
        info_path=INPUT_SHHS_INFO,
        id_col="nsrrid",
        ahi_col="nsrr_ahi_hp3r_aasm15",
        fs_col="fs",
        method= "biosppy", # ECG from SHHS
    )

    INPUT_MESA_PATH = ROOT_PATH / "data/processed/MESA_PPG"
    INPUT_MESA_INFO = ROOT_PATH / "data/raw/MESA/datasets/mesa-sleep-harmonized-dataset-0.8.0.csv"

    process_dataset(
        dataset_name="MESA",
        in_dir=INPUT_MESA_PATH,
        out_dir=OUTPUT_PATH,
        info_path=INPUT_MESA_INFO,
        id_col="mesaid",
        ahi_col="nsrr_ahi_hp3u",
        fs_col="fs",
        method="neurokit", # PPG from MESA
    )

    logging.info("All done! SHHS + MESA IBI files are now in the same folder.")
