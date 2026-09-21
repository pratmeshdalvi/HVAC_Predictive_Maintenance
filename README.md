# 🏢 HVAC & Industrial Equipment Predictive Maintenance using Deep Learning

This repository contains a Google Colab notebook implementing a predictive maintenance and failure detection system for HVAC and industrial equipment. The project is inspired by the research paper:

> **"Application of deep learning in facility management and maintenance for heating, ventilation, and air conditioning"**  
> *Sanzana et al. (2022), Automation in Construction.*

---

## 📌 Project Overview

Predictive maintenance (PdM) leverages real-time IoT sensor telemetry to forecast machine breakdowns before they occur. This project compares three supervised Deep Learning architectures (**MLP**, **LSTM**, **1D-CNN**) and one unsupervised **Autoencoder** anomaly detection model on the **AI4I 2020 Predictive Maintenance Dataset** (UCI ML Repository).

---

## 🛠️ Key Features

- **Automated Setup & Ingestion**: Loads the dataset directly from UCI ML Repository without requiring manual uploads.
- **Exploratory Data Analysis (EDA)**: Class balance diagnostics (~3.39% failures) and physical sensor correlation heatmaps.
- **Preprocessing Pipeline**: Categorical encoding, `StandardScaler` normalization, stratified 80/20 train/test split, cost-sensitive class weighting, and 2D/3D tensor reshaping.
- **Supervised Deep Learning Benchmark**:
  - **MLP (Multi-Layer Perceptron)**: Baseline feedforward network with Batch Normalization and Dropout.
  - **LSTM (Long Short-Term Memory)**: Recurrent sequence network capturing state gating and temporal representations.
  - **1D-CNN (1D Convolutional Neural Network)**: Spatial filter extractor across sensor input channels.
- **Unsupervised Anomaly Detection**: Deep Autoencoder trained strictly on normal baseline operational telemetry ($y=0$) using reconstruction error thresholding.
- **Explainable AI (SHAP)**: Game-theoretic feature attributions to highlight top failure predictors for facility managers (Torque, Tool wear, Thermal differential).

---

## 📊 Benchmark Performance Summary

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MLP Baseline** | 0.9017 | 0.2394 | 0.8235 | 0.3709 | 0.9552 |
| **LSTM Recurrent** | 0.8850 | 0.2185 | 0.8529 | 0.3473 | 0.9421 |
| **1D-CNN Spatial** | 0.8848 | 0.2222 | 0.8824 | 0.3550 | **0.9636** |
| **Autoencoder (Anomaly)** | 0.9255 | 0.2346 | 0.5588 | 0.3304 | 0.8504 |

---

## 🚀 Getting Started

1. Open [`HVAC_Predictive_Maintenance_DL.ipynb`](HVAC_Predictive_Maintenance_DL.ipynb) in [Google Colab](https://colab.research.google.com).
2. Enable GPU acceleration (`Runtime` > `Change runtime type` > `T4 GPU`).
3. Click `Runtime` > `Run all` to execute the full pipeline end-to-end.

---

## 📜 References & Acknowledgments

- **Paper**: Sanzana, M. R., Maul, T., Wong, J. Y., Abdulrazic, M. O. M., & Yip, C. C. (2022). Application of deep learning in facility management and maintenance for heating, ventilation, and air conditioning. *Automation in Construction*, 141, 104445.
- **Dataset**: AI4I 2020 Predictive Maintenance Dataset, UCI Machine Learning Repository.
