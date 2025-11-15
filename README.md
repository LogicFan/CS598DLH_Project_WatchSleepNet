# CS 598 Deep Learning for Health Final Project

## Overview

This project is a replication of the paper **WatchSleepNet: A Novel Model and Pretraining Approach for Advancing Sleep Staging with Smartwatches** by Wang et al. (2025). The original paper introduces a novel deep learning model for sleep staging using data from smartwatches, leveraging pretraining techniques to improve performance.

The goal of this replication is to implement and validate the WatchSleepNet model, potentially extending it with additional experiments or optimizations.

## Structure

```
.
├── README.md
├── requirements.txt
├── data/                  # See Data Preparation below
├── prepare_data/
│   ├── DREAMT_PIBI_SE.py  # Script to extract IBI from downloaded DREAMT dataset
│   ├── MESA_PPG.py        # Script to process MESA data
│   ├── SHHS_ECG.py        # Script to process SHHS ECG data
│   ├── SHHS_MESA_IBI.py   # Script to process combined SHHS and MESA IBI data
│   └── utils.py           # Utility functions
├── modeling/
│   ├── check_corrupted_files.py # list or remove corrupted .npz data files
│   ├── check_corrupted.sh # run check_corrupted_files.py on all .npz data files with options
│   ├── config.py # Configure dataset and model params
│   ├── data_setup.py
│   ├── engine.py
│   ├── insightsleepnet_hpt.py # Run program to perform hyperparameter tuning for InsightSleepNet
│   ├── sleepconvnet_hpt.py # Run program to perform hyperparameter tuning for SleepConvNet
│   ├── watchsleepnet_hpt.py # Run program to perform hyperparameter tuning for WatchSleepNet
│   ├── models # Model architectures
│   │   ├── insightsleepnet.py
│   │   ├── sleepconvnet.py
│   │   └── watchsleepnet.py
│   ├── train_cv.py
│   ├── train_transfer.py # Run program to perform transfer learning experiments
│   ├── watchsleepnet_cv_ablation.py # Run program to perform ablation experiments (DREAMT) on WatchSleepNet
│   └── watchsleepnet_transfer_ablation.py # Run program to perform ablation experiments (Transfer Learning) on WatchSleepNet
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

1. Download the following dataset
- [Dataset for Real-time sleep stage EstimAtion using Multisensor wearable Technology](https://physionet.org/content/dreamt/2.0.0/)
- [Multi-Ethnic Study of Atherosclerosis](https://sleepdata.org/datasets/mesa)
- [Sleep Heart Health Study](https://sleepdata.org/datasets/shhs)

2. Put the corresponding dataset in the `raw` folder.

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
        ├── SHHS_EEG        # Processed data for the SHHS dataset
        └── SHHS_MESA_IBI   # Combined IBI data from SHHS and MESA
```

3. Run the following commands

```bash
python prepare_data/DREAMT_PIBI_SE.py
python prepare_data/MESA_PPG.py
python prepare_data/SHHS_EEG.py
python prepare_data/SHHS_MESA_IBI.py
```

### Training

1. To check all datasets for corrupted files:
```bash
python modeling/check_corrupted_files.py --dataset all
```

Remove corrupted files using:
```bash
python modeling/check_corrupted_files.py --fix
```

2. Use transfer learning or train model from scratch (see below headers)

#### Transfer Learning

You can perform transfer learning experiments (pre-train on IBI from SHHS+MESA and test on DREAMT IBI) using the `modeling/train_transfer.py`. Run the experiment with WatchSleepNet:
```
python train_transfer.py --model=watchsleepnet
```
To perform the experiment with other benchmark models (i.e. InsightSleepNet, SleepConvNet), indicate selected model using the `--model` parser argument:
```
python train_transfer.py --model=insightsleepnet
```
```
python train_transfer.py --model=sleepconvnet
```

#### **No Transfer Learning Cross Validation:**

Train models from scratch on a single dataset (faster training)
To train on DREAMT IBI data without pretraining on SHHS+MESA IBI data, use the `modeling/train_cv.py`
```bash
python train_cv.py --model=watchsleepnet --train_dataset=dreamt_pibi
```
### Hyperparameter Tuning

You can perform hyperparameter tuning for WatchSleepNet, InsightSleepNet, and SleepConvNet. For example, to tune WatchSleepNet run
```
python watchsleepnet_hpt.py
```

#### TODO: DESCRIBE ABLATION EXPERIMENTS [PIKKIN]

### Evaluation


#### 🎯 **Evaluation Methods**

#### **Method 1: Using the --testing Flag (Skip Retraining)**

If you've already trained a model and want to skip the pretraining phase, use the `--testing` flag:

```bash
cd ./WatchSleepNet/modeling

# Evaluate WatchSleepNet (loads pretrained checkpoint, only runs fine-tuning evaluation)
python train_transfer.py --model=watchsleepnet --testing

# Evaluate InsightSleepNet
python train_transfer.py --model=insightsleepnet --testing

# Evaluate SleepConvNet
python train_transfer.py --model=sleepconvnet --testing

# Evaluate SleepPPGNet
python train_transfer.py --model=sleepppgnet --testing
```


#### **Method 2: Standalone Evaluation Script**

Use the new `evaluate_model.py` script for quick evaluations on any checkpoint:

#### **Basic Usage:**

```bash
cd /Users/leeaaron/Desktop/UIUC/WatchSleepNet/modeling

# Evaluate WatchSleepNet on DREAMT (fold 1)
python evaluate_model.py \
    --model watchsleepnet \
    --dataset dreamt_pibi \
    --checkpoint checkpoints/watchsleepnet/dreamt_pibi/best_saved_model_ablation_separate_pretraining_fold1.pt

# Evaluate on SHHS+MESA pretraining dataset
python evaluate_model.py \
    --model watchsleepnet \
    --dataset shhs_mesa_ibi \
    --checkpoint checkpoints/watchsleepnet/shhs_mesa_ibi/best_saved_model_ablation_separate_pretraining.pt
```

#### **With AHI Breakdown:**

Evaluate performance across different sleep apnea severity categories:

```bash
python evaluate_model.py \
    --model watchsleepnet \
    --dataset dreamt_pibi \
    --checkpoint checkpoints/watchsleepnet/dreamt_pibi/best_saved_model_ablation_separate_pretraining_fold1.pt \
    --ahi_breakdown
```

**AHI Categories:**
- **Normal**: AHI < 5 (no sleep apnea)
- **Mild**: 5 ≤ AHI < 15 (mild sleep apnea)
- **Moderate**: 15 ≤ AHI < 30 (moderate sleep apnea)
- **Severe**: AHI ≥ 30 (severe sleep apnea)


#### **Ablation**

This ablation replaces TCN in SleepConvNet with Depthwise Separable Convolutions (DS-TCN) which reduces params & FLOPs.

Hypothesis: DS-TCN retains temporal receptive field with fewer parameters, improving generalization on small DREAMT and inference speed on smartwatches.

Command to modify the dilation block of SleepConvNet into DS-TCN:

```bash
cd modeling
python train_transfer.py --model sleepconvnet --ablation True
```

<!--
To evaluate the trained model:
```bash
python evaluate.py --model_path checkpoints/model.pth --data_path data/test/
```
<!--
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
