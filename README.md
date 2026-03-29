# ✈️ Aircraft Identification Using Deep Learning

## 📌 Overview

Aircraft identification is a challenging computer vision problem due to significant variability in real-world conditions such as viewpoint, lighting, scale, and background clutter. This project develops a deep learning-based system to classify aircraft images while addressing these challenges, with a particular focus on **multi-angle robustness**.

The system leverages convolutional neural networks (CNNs) and transfer learning techniques to improve classification performance across diverse visual conditions.

---

## 🎯 Objectives

* Build a diverse dataset of aircraft images from multiple viewpoints
* Implement a **baseline CNN model** trained from scratch
* Apply **transfer learning (ResNet50, EfficientNet)**
* Analyze the impact of viewpoint diversity on classification accuracy
* Evaluate performance using standard classification metrics

---

## 🧠 Approach

### 1. Data Collection

Datasets are collected from:

* Kaggle (Military Aircraft Detection Dataset)
* FGVC Aircraft Dataset
* Public web sources (Google Images)

The dataset includes:

* Military aircraft
* Commercial aircraft
* Propeller planes
* Jet aircraft

---

### 2. Data Preprocessing

All images undergo:

* Resizing to **224 × 224**
* Normalization to [0,1]
* Data augmentation:

  * Rotation
  * Horizontal flipping
  * Brightness variation

---

### 3. Model Architectures

#### 🔹 Baseline Model

A custom CNN trained from scratch:

* Convolutional layers
* ReLU activations
* Max pooling
* Fully connected layers

#### 🔹 Transfer Learning Models

* ResNet50
* EfficientNet

These models use pretrained ImageNet weights to improve generalization and reduce training time.

---

### 4. Training Pipeline

1. Load and preprocess dataset
2. Split into:

   * 70% Training
   * 15% Validation
   * 15% Testing
3. Train model using mini-batch gradient descent
4. Validate performance each epoch
5. Test final model on unseen data

---

## 📊 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

## 🗂️ Project Structure

```
Aircraft-Identification/
│
├── data/
│   ├── raw/                # Original datasets
│   ├── processed/          # Cleaned and resized images
│
├── notebooks/              # Jupyter notebooks for exploration
│   ├── exploration.ipynb
│
├── src/
│   ├── data_loader.py      # Dataset loading
│   ├── preprocessing.py    # Image transformations
│   ├── model.py            # CNN + transfer models
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│
├── results/
│   ├── plots/              # Graphs and visualizations
│   ├── metrics/            # Saved evaluation metrics
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## ⚙️ Installation

Clone the repository and install dependencies:

```
git clone https://github.com/your-username/Aircraft-Identification.git
cd Aircraft-Identification

python -m venv venv
source venv/bin/activate      # Windows: .\venv\Scripts\activate

pip install -r requirements.txt
```

---

## ▶️ Usage

### Train the Model

```
python src/train.py
```

### Evaluate the Model

```
python src/evaluate.py
```

---

## 📈 Expected Results

* Improved accuracy using transfer learning
* Stronger generalization across viewpoints
* Better robustness to real-world conditions

---

## 🚧 Project Status

* [x] Proposal completed
* [ ] Dataset collection
* [ ] Data preprocessing
* [ ] Baseline CNN implementation
* [ ] Transfer learning models
* [ ] Training and evaluation
* [ ] Results analysis

---

## 👥 Team Members

* Alexander Bernal
* Aaron Majdali
* Tess Breckenridge
* Artip Nakchinda

---

## 📚 References

* FGVC Aircraft Dataset
* Military Aircraft Detection Dataset (Kaggle)
* PyTorch Documentation
* ImageNet Pretrained Models

---

## 🔮 Future Work

* Multi-task learning (aircraft + viewpoint classification)
* Object detection (YOLO, Faster R-CNN)
* Real-time aircraft recognition system

---

## 📄 License

This project is intended for academic use.
