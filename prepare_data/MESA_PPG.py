import os
from tqdm import tqdm
from utils import read_edf_data, save_to_npz
from pathlib import Path

def process_mesa(in_path, out_path):
    os.makedirs(out_path, exist_ok=True)
    mesa_dir = in_path / "polysomnography/edfs/"
    files = [f for f in os.listdir(mesa_dir) if f.endswith(".edf")]
    for file in tqdm(files):
        sid = file.split("-")[-1].split(".")[0]
        data_path = mesa_dir / file
        label_path = in_path / "polysomnography/annotations-events-profusion" / f"{file.split('.')[0]}-profusion.xml"
        try:
            data, fs, stages = read_edf_data(data_path, label_path, dataset="MESA", select_chs=["Pleth"])
            save_to_npz(f"{out_path}/mesa-{sid}.npz", data, stages, fs)
        except Exception as e:
            print(f"Error processing {sid}: {e}")

if __name__ == "__main__":
    ROOT_PATH = Path(__file__).resolve().parent.parent
    IN_PATH = ROOT_PATH / "data/raw/MESA"
    OUT_PATH = ROOT_PATH / "data/processed/MESA_PPG"
    process_mesa(IN_PATH, OUT_PATH)
