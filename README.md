# DoMiNoMatch: A Unified Framework for Multi-Target Semi-Supervised Domain Adaptation

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Pretrained Weights](#pretrained-weights)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

**DoMiNoMatch** is a unified framework designed for Multi-Target Semi-Supervised Domain Adaptation (MT-SSDA) under limited annotations. By combining Fourier-based domain mixing with a novel Adaptation Network, DoMiNoMatch addresses the challenges of data scarcity, diverse visual styles, and severe distribution shifts in open-world visual recognition. 

This repository contains the official implementation of our paper, seamlessly integrated into the Microsoft Unified Semi-supervised learning Benchmark (USB) framework. Our core algorithm contribution is located at `semilearn/algorithms/dominomatch/dominomatch.py`.
### Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Target SSDA** | Effectively adapts models to multiple target domains simultaneously using limited labeled data. |
| **FourierMix Domain Bridge** | Facilitates cross-domain style adaptation by mixing source and target domain features in the frequency domain. |
| **Robust Architecture** | Integrates consistency regularization, pseudo-labeling, and contrastive distillation, built upon the scalable Microsoft USB framework. |
| **Dataset Support** | Automated scripts to download and format widely published datasets: PACS and DomainNet. |

---

## Installation

### Requirements

- Python 3.8+
- PyTorch
- CUDA

### Setup Environment

```bash
# 1. Clone the repository
git clone https://github.com/kmstorm/DoMiNoMatch.git
cd DoMiNoMatch

# 2. Create conda environment
conda create -n dominomatch python=3.8 -y
conda activate dominomatch

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Dataset Preparation

### Supported Datasets

| Dataset | Domains | Download Script |
|---------|---------|-----------------|
| **PACS** | Art Painting, Cartoon, Photo, Sketch | `data/pacs/download_pacs.sh` |
| **DomainNet** | Clipart, Infograph, Painting, Quickdraw, Real, Sketch | `data/domainnet/download_domainnet.sh` |

### Download Instructions

Raw images are excluded from version control. We provide bash scripts to download and format the datasets automatically. 

**Requirements:** `wget`, `unzip`

```bash
# Download and prepare PACS dataset (~500MB)
cd data/pacs
bash download_pacs.sh

# Download and prepare DomainNet dataset (~80GB)
cd ../domainnet
bash download_domainnet.sh
```
*For more details on the directory structure and train/test splits, please refer to [`data/README.md`](data/README.md).*

---

## Usage

### Training

You can start training the DoMiNoMatch model using the standard USB training script with your specific configuration. Example usage:

```bash
conda activate dominomatch

# Train DoMiNoMatch (adjust the config path accordingly)
python train.py --c config/dominomatch/dominomatch_pacs.yaml
```

### Evaluation

After training, evaluate the model on the target domain test sets:

```bash
python eval.py --dataset pacs --load_path /PATH/TO/CHECKPOINT
```

---

## Pretrained Weights

> [!NOTE]
> Pretrained weights will be published after the paper is accepted.

---

## Citation

If you find this project or our code helpful for your research, please consider citing our paper:

```bibtex
@article{dominomatch,
  title={DoMiNoMatch: A Unified Framework for Multi-Target Semi-Supervised Domain Adaptation},
  author={Minh Bao Kha, Tuan Nam Do, Ngoc Minh Nguyen, Tuan Linh Dang},
  journal={Submitted to The Visual Computer},
  year={2026}
}
```
*(Citation link will be updated after acceptance)*

---

## Acknowledgements

This project builds upon the knowledge and codebase of:

- [Microsoft Semi-supervised-learning (USB)](https://github.com/microsoft/Semi-supervised-learning): A Unified Semi-supervised learning Benchmark for Classification.
