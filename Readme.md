
# 🚀 LoopVerse 2025 – EuroSAT Land Cover Classification 🌍

## 🎯 Problem Statement

Satellite imagery is vital for environmental monitoring, urban planning, and disaster response. The **EuroSAT dataset** contains multispectral satellite image tiles classified into 10 land-use categories like Forest, Urban, Water, and Agriculture. Your mission: build a robust **CNN model from scratch** to accurately classify these land-cover types despite noisy and inconsistent raw data.

---

## 🔥 Project Highlights

### 🧹 1. Data Cleaning & Preprocessing

* Handled mixed image formats: `.tif`, `.png`, `.jpg`
* Removed corrupted images (black patches, noise)
* Normalized image size & channels to **224×224 RGB**
* Applied **Gaussian filtering** for noise reduction

### 🎨 2. Feature Extraction & Visualization

* Extracted low-level features (color histograms, edges)
* Visualized features to confirm better class separability post-cleaning

### 🏗️ 3. CNN Model Design (Built from Scratch)

* **4 convolutional layers** (3×3 kernels), with BatchNorm + ReLU
* MaxPooling layers for downsampling
* Dropout layers to prevent overfitting
* Fully connected dense layers for final classification
* Training on an 80/20 split with training & validation accuracy/loss graphs

### ⚙️ 4. Performance Improvements

* Data augmentation: random rotations, flips, brightness/contrast jitter ✨
* Adam optimizer with weight decay
* Learning rate scheduling with **ReduceLROnPlateau** ⬇️
* Early stopping to avoid overfitting 🚦

### 📊 5. Model Evaluation

* Confusion matrix showing per-class performance 🔍
* Reported overall accuracy, precision, recall, and F1-score 🏅
* Saved predictions CSV with columns: `image_id`, `predicted_label` 💾

### 🔎 6. Analysis & Explanation

* Compared baseline vs. improved CNN models
* Demonstrated impact of data cleaning and augmentation on accuracy (improved from \~85% to >90%)
* Visualized learned feature maps and activation layers 🖼️

---

## 🚀 How to Run

1. Clone the repo
2. Install dependencies:

   ```bash
   pip install torch torchvision matplotlib seaborn scikit-learn pandas
   ```
3. Place cleaned EuroSAT dataset in `cleaned_dataset/EuroSAT_RGB`
4. Run training script:

   ```bash
   python train.py
   ```
5. Run evaluation script:

   ```bash
   python evaluate.py
   ```

---


Feel free to reach out for questions or collaborations. 🙌

---

