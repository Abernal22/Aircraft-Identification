#  Military Aircraft Identification Using Deep Learning

## Overview

This project focuses on **fine-grained visual classification (FGVC)** of aircraft using deep learning. The goal is to develop a robust computer vision system capable of identifying aircraft under **real-world conditions**, including variations in:

* Viewpoint (front, side, top, diagonal)
* Lighting and weather
* Background clutter
* Image resolution

The project specifically targets **military aircraft recognition**, where models must distinguish between visually similar aircraft using subtle structural features rather than color or texture.

---

## Objectives

* Build a complete deep learning pipeline for aircraft classification
* Investigate the impact of **multi-angle image data** on model performance
* Compare multiple architectures:

  * Custom CNN (baseline)
  * ResNet50 (transfer learning)
  * EfficientNet-B0
  * Vision Transformer (ViT)
* Evaluate trade-offs between **accuracy and computational efficiency**
* Improve robustness using **advanced data augmentation techniques**

---

## Key Features

* Multi-angle dataset design
* Advanced augmentation:

  * Random Erasing
  * CutMix (planned)
  * MixUp (planned)
* Modular PyTorch pipeline
* Config-driven training
* Evaluation using:

  * Accuracy
  * Precision / Recall
  * F1-score
  * Confusion Matrix

---

## Project Structure

```
Aircraft-Identification/
│
├── configs/                # Configuration files (hyperparameters)
│   └── config.yaml
│
├── data/
│   └── processed/          # Train / Val / Test datasets
│       ├── train/
│       ├── val/
│       └── test/
│
├── models/                 # Saved model weights
│
├── notebooks/              # Experiments and visualization
│
├── outputs/                # Results (plots, metrics, confusion matrices)
│
├── src/                    # Core source code
│   ├── augmentations.py    # Data augmentation pipeline
│   ├── data_loader.py      # Dataset loading
│   ├── evaluate.py         # Model evaluation
│   ├── model.py            # Model definitions
│   ├── train.py            # Training script
│   └── utils.py            # Utility functions
│
├── README.md
└── requirements.txt
```

---

## Dataset

The dataset is constructed from publicly available aircraft images, including the:

* Military Aircraft Detection Dataset (Kaggle)

Classes (current baseline):

* `military`
* `non_military`

Future extension:

* F-16
* F/A-18
* C-130
* A-10

Dataset split:

* **70% Training**
* **15% Validation**
* **15% Testing**

---

##  Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Aircraft-Identification.git
cd Aircraft-Identification
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

### 1. Train the Model

```bash
python src/train.py
```

This will:

* Load dataset
* Train the model
* Save weights to `/models`

---

### 2. Evaluate the Model

```bash
python src/evaluate.py
```

This will output:

* Confusion matrix
* Precision / Recall / F1-score

---

## ⚙️ Configuration

All hyperparameters are controlled via:

```
configs/config.yaml
```

Example:

```yaml
batch_size: 32
learning_rate: 0.001
epochs: 10
image_size: 224
num_classes: 2
model_name: cnn
```

---

## Models

### Baseline

* Custom CNN (implemented)

### Transfer Learning (Planned / In Progress)

* ResNet50
* EfficientNet-B0

### Advanced Model (Planned)

* Vision Transformer (ViT)

---

##  Evaluation Metrics

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

These metrics help analyze performance under **real-world variability and class imbalance**.

---

##  Research Focus

This project investigates:

* Viewpoint invariance in deep learning
* Impact of dataset diversity
* Overfitting to background vs aircraft features
* Model efficiency vs accuracy trade-offs

---

##  Future Work

* Implement CutMix and MixUp augmentation
* Add ResNet50 and EfficientNet training
* Introduce fine-grained aircraft classification
* Hyperparameter tuning
* GPU acceleration
* Deployment as a web application

---

## Authors

* Alexander Bernal
* Aaron Majdali
* Tess Breckenridge
* Artip Nakchinda

University of New Mexico

---

## License

This project is for academic and research purposes.

---

## References

* Kaggle Military Aircraft Dataset
* PyTorch Documentation
* ImageNet Pretrained Models
* Research papers on FGVC and aircraft classification

---
