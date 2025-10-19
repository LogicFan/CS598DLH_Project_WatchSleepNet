import os
from tqdm import tqdm
from pathlib import Path
from utils import read_edf_data, save_to_npz

def process_shhs(in_path, out_path):
    shhs_dirs = [
        in_path / "polysomnography/edfs/shhs1/",
        in_path / "polysomnography/edfs/shhs2/"
    ]
    for shhs_dir in shhs_dirs:
        # Extract the folder name (shhs1 or shhs2)
        dir_label = shhs_dir.name
        files = [f for f in shhs_dir.iterdir() if f.suffix == '.edf']
        for file in tqdm(files):
            sid = file.stem.split("-")[1]
            data_path = file
            label_path = in_path / "polysomnography/annotations-events-profusion" / dir_label / f"{file.stem}-profusion.xml"
            try:
                data, fs, stages = read_edf_data(data_path, label_path, dataset="SHHS", select_chs=["ECG"])
                save_to_npz(out_path / f"{dir_label}-{sid}.npz", data, stages, fs)
            except Exception as e:
                print(f"Error processing {sid}: {e}")

if __name__ == "__main__":
    ROOT_PATH = Path(__file__).resolve().parent.parent
    IN_PATH = ROOT_PATH / "data/raw/SHHS"
    OUT_PATH = ROOT_PATH / "data/processed/SHHS_ECG"
    os.makedirs(OUT_PATH, exist_ok=True)
    process_shhs(IN_PATH, OUT_PATH)

