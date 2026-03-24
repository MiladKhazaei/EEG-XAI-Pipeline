# 🧠 EEG-XAI-Pipeline: Two-Pass Architecture for High-Performance Deep Learning Visualization

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C)
![SHAP](https://img.shields.io/badge/Explainable%20AI-SHAP-brightgreen)

## 📝 Overview

Generating Explainable AI (XAI) visualizations like **Grad-CAM** and **SHAP** for large time-series datasets (e.g., EEG signals) is computationally expensive. Running visual rendering functions inside a standard evaluation loop for thousands of samples quickly leads to **Memory Leaks (OOM)** and severe I/O bottlenecks.

This repository provides an optimized **Two-Pass Architecture** that decouples mathematical metadata extraction from visual rendering, reducing processing time from hours to minutes while preventing system crashes.

---

## 🏗️ Architecture

Instead of plotting every epoch or batch, this pipeline uses a targeted approach:

### 1️⃣ Pass 1: Computation (`pass1_computation.py`)

- Skips visual rendering entirely.
- Uses a computationally optimized baseline for `shap.GradientExplainer`.
- Mathematically extracts Confidence, Localized Channel Variance, and Peak Intensity for each window.
- Outputs a lightweight `data_driven_window_metadata.csv` containing the statistical profile of the model's decision-making process.

### 2️⃣ Pass 2: Visualization (`pass2_visualization.py`)

- Parses the CSV to identify the "Best IDs" (the highest-confidence, clinically relevant true positives/negatives).
- Fast-forwards through the dataset and **only** triggers the heavy `matplotlib` and SHAP visualization functions for those specific, highly-representative IDs.

---

## 📊 Visual Results

_Targeted extraction of pre-ictal transients and artifacts._

_(Note: Replace the URL below with the actual link to your best image once uploaded to GitHub)_
![Representative XAI Output](https://github.com/MiladKhazaei/EEG-XAI-Pipeline/blob/main/best_gradcam_result.png?raw=true)

---

## 🚀 Usage

### Step 1: Run the Computation Pass

Extract mathematical metadata without plotting:

```bash
python pass1_computation.py
```

### Step 2: Filter the Data

Open the generated data_driven_window_metadata.csv. Filter and identify the sample_idx of the most representative windows based on your clinical narrative (e.g., temporal lobe focus for seizures).

### Step 3: Run the Visualization Pass

Update the final_target_ids array inside pass2_visualization.py with your selected IDs, then run:

```bash
python pass2_visualization.py
```
