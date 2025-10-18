# CS 598 Deep Learning for Health Final Project

## Overview

This project is a replication of the paper **WatchSleepNet: A Novel Model and Pretraining Approach for Advancing Sleep Staging with Smartwatches** by Wang et al. (2025). The original paper introduces a novel deep learning model for sleep staging using data from smartwatches, leveraging pretraining techniques to improve performance.

The goal of this replication is to implement and validate the WatchSleepNet model, potentially extending it with additional experiments or optimizations.

## Structure

```
.
├── prepare_data
│   ├── dreamt.py # Run program to program to extract IBI from downloaded DREAMT dataset.
├── README.md
└── requirements.txt
```

## Installation

### Prerequisites

- Python 3.10
- PyTorch 2.9
- Other dependencies (see `requirements.txt`)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/LogicFan/CS598DLH_Project_WatchSleepNet.git
   cd CS598DLH_Project_WatchSleepNet
   ```

2. Create Conda environment:
   ```bash
   conda create -n <env_name> python=3.10
   conda activate <env_name>
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Preparation

Download the following dataset
- [Dataset for Real-time sleep stage EstimAtion using Multisensor wearable Technology](https://physionet.org/content/dreamt/2.0.0/)
- [Multi-Ethnic Study of Atherosclerosis](https://sleepdata.org/datasets/mesa)
- [Sleep Heart Health Study](https://sleepdata.org/datasets/shhs)

and put the corresponding dataset in the `raw` folder.

```
.
└── data
    ├── raw
    │   ├── DREAMT  # Raw data for the DREAMT dataset
    │   ├── MESA    # Raw data for the MESA dataset
    │   └── SHHS    # Raw data for the SHHS dataset
    └── processed
        ├── DREAMT_PIBI_SE  # Processed data for the DREAMT dataset
        ├── MESA_PPG        # Processed data for the MESA dataset
```

and run the following commands

```bash
python prepare_data/DREAMT_PIBI_SE
python prepare_data/MESA_PPG.py
```

<!--
### Training

To train the model:
```bash
python train.py --config config.yaml
```

### Evaluation

To evaluate the trained model:
```bash
python evaluate.py --model_path checkpoints/model.pth --data_path data/test/
```

## Dataset

This project uses publicly available sleep staging datasets such as:
- TODO

Ensure compliance with dataset licenses and usage terms.

## Model Architecture

TODO

## Results

TODO

-->

## Citation

If you use this code, please cite the original paper:

```
@article{wang2025watchsleepnet,
  title={WatchSleepNet: A Novel Model and Pretraining Approach for Advancing Sleep Staging with Smartwatches},
  author={Wang, Will and others},
  journal={Proceedings of Machine Learning Research},
  volume={287},
  pages={1--15},
  year={2025}
}
```

## Acknowledgement

This implementation is based on the code from the [WatchSleepNet_public repository](https://github.com/WillKeWang/WatchSleepNet_public).

## Authors

[Yongda Fan](mailto:yongdaf2@illinois.edu), [Pikkin Lau](mailto:pikkinl2@illinois.edu), [Aaron Lee](mailto:aaroncl2@illinois.edu)