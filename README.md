# Rice Type Classification with PyTorch

A binary neural network that classifies rice grains as **Jasmine** or **Gonen** from morphological features — achieving **98.79% test accuracy**. Built end-to-end in PyTorch with feature normalization, custom datasets, GPU training, and evaluation.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-MLP-EE4C2C)
![License](https://img.shields.io/badge/License-MIT-green)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-98.79%25-brightgreen)

---

## Highlights

- Binary classification: **Gonen (0)** vs **Jasmine (1)**
- 18,185 samples with 10 morphological features
- Min-max feature normalization before training
- Lightweight MLP (~241 parameters) trained from scratch
- ~70 / 15 / 15 train–validation–test split
- GPU-accelerated training (CUDA)
- Training curves for loss and accuracy across 10 epochs

---

## Results

| Metric | Value |
| --- | --- |
| Test accuracy | **98.79%** |
| Peak validation accuracy | **98.86%** (epoch 9) |
| Final training accuracy | 98.37% |
| Epochs | 10 |
| Parameters | 241 |

The model jumps to ~95% validation accuracy by epoch 1 and stays tightly aligned between train and validation metrics — a strong sign of good generalization with no clear overfitting.

---

## Architecture

Tabular features (10) → fully connected network with sigmoid output for binary classification:

| Layer | Details |
| --- | --- |
| Linear | 10 → 20 hidden units |
| Linear | 20 → 1 |
| Sigmoid | probability in [0, 1] |

**Training setup:** Adam (`lr=0.0003`), BCELoss, batch size 32, 10 epochs.

**Decision rule:** prediction rounded to `0` (Gonen) or `1` (Jasmine).

---

## Features

| Feature | Description |
| --- | --- |
| Area | Grain area |
| MajorAxisLength | Major axis length |
| MinorAxisLength | Minor axis length |
| Eccentricity | Ellipse eccentricity |
| ConvexArea | Convex hull area |
| EquivDiameter | Equivalent diameter |
| Extent | Ratio of pixels in bounding box |
| Perimeter | Grain perimeter |
| Roundness | Shape roundness |
| AspectRation | Aspect ratio |

---

## Project structure

```text
Rice Classification Pytorch/
├── rice_classification.ipynb         # Full training & evaluation pipeline
├── rice_classification_model.pt      # Saved model weights
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

The Kaggle dataset (`rice-type-classification/`) is downloaded at runtime and is not committed to the repo.

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/Tech-Watt/rice-classification-pytorch.git
cd rice-classification-pytorch
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Install a CUDA-enabled PyTorch build from [pytorch.org](https://pytorch.org/get-started/locally/) if you have an NVIDIA GPU.

### 4. Kaggle credentials

The notebook downloads the dataset with [`opendatasets`](https://github.com/JovianML/opendatasets). You’ll need a free Kaggle account and API token (`kaggle.json`) when prompted on first run.

---

## Usage

### Train / evaluate

Open and run the notebook:

```bash
jupyter notebook rice_classification.ipynb
```

The notebook covers:

1. Dataset download from Kaggle  
2. EDA and feature inspection  
3. Min-max normalization  
4. Train / validation / test split  
5. Custom `Dataset` + `DataLoader`  
6. MLP definition and `torchsummary`  
7. Training loop with loss/accuracy logging  
8. Test-set evaluation and training curves  
9. Single-sample prediction  

### Inference example

```python
import torch

# features must be min-max normalized the same way as training
x = torch.tensor([...], dtype=torch.float32)  # shape: (10,)

model.eval()
with torch.no_grad():
    prob = model(x.to(device)).item()
    pred = round(prob)  # 0 = Gonen, 1 = Jasmine

print("Jasmine" if pred == 1 else "Gonen", f"(p={prob:.4f})")
```

---

## Tech stack

- **Python** — core language  
- **PyTorch** — model, training, GPU acceleration  
- **NumPy / Pandas** — data handling  
- **scikit-learn** — train/test split  
- **Matplotlib** — training curves  
- **opendatasets** — Kaggle download  
- **torchsummary** — architecture inspection  

---

## Dataset

[Rice Type Classification](https://www.kaggle.com/datasets/mssmartypants/rice-type-classification) — 18,185 rice grain samples labeled as Jasmine (`1`) or Gonen (`0`), derived from morphological measurements.

---

## What I learned / demonstrated

- Building a binary MLP classifier in PyTorch (`nn.Module` + BCELoss)  
- Feature scaling and tabular data pipelines with custom datasets  
- Train / validation / test evaluation to monitor generalization  
- Interpreting loss and accuracy curves for overfitting checks  
- End-to-end ML workflow: data → preprocess → train → evaluate → predict  

---

## License

This project is licensed under the [MIT License](LICENSE).
