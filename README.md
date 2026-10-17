# Currency Note Recognition Assistant

An end-to-end Computer Vision system for automatically recognizing and classifying currency banknotes (Afghan Afghani and Pakistani Rupee) and their denominations using deep transfer learning with ResNet-50 and a modern dual-deployment architecture (Streamlit local interface and Hugging Face + Netlify cloud production).

---

## Project Overview

Identifying currency notes manually can be challenging in various real-world scenarios. This system automates the recognition of different banknote denominations accurately using deep learning, with potential applications in financial kiosks, automated vending machines, and accessibility tools for the visually impaired.

### 3-Stage Hierarchical Classification
1. **Stage 1 (Banknote Verification):** Uses OpenCV contour analysis, confidence thresholding (40.0%), and class separation margin to verify whether the image contains a genuine banknote or an out-of-distribution object (e.g., vehicles, furniture, animals).
2. **Stage 2 (Country Identification):** Aggregates class probability masses to determine whether the currency belongs to the Afghan Afghani (AFN) or Pakistani Rupee (PKR) family.
3. **Stage 3 (Denomination Recognition):** Pinpoints the exact note value within the confirmed national currency family.

### Supported Currencies & Denominations
* **Afghan Afghani (AFN):** 10, 20, 50, 100, 500, 1000
* **Pakistani Rupee (PKR):** 10, 20, 50, 100, 500, 1000, 5000

---

## Project Architecture & Folder Structure

```text
currency_detection/
├── app.py                # Interactive Streamlit Web Application (Local Interface)
├── train.py              # Transfer learning and deep fine-tuning script (ResNet-50)
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation & instructions
├── dataset/              # Banknote dataset
│   ├── Afghani/          # Subdirectories for each note denomination
│   └── PKR/              # Subdirectories for each note denomination
├── models/               # Saved model weights & mappings
│   ├── model.pth         # Fine-tuned ResNet-50 model weights
│   └── class_names.json  # Class index mapping
├── utils/                # Helper modules
│   ├── __init__.py
│   ├── dataset_loader.py # PyTorch Dataset & data augmentation pipelines
│   └── predict.py        # 3-stage hierarchical prediction engine & OpenCV contour extractor
├── hf_space_backend/     # Cloud Backend for Hugging Face Spaces (Gradio + ZeroGPU)
└── netlify_frontend/     # Cloud Frontend for Netlify (HTML5, CSS3, JavaScript)
```

---

## Installation & Setup

### 1. Clone or Open the Repository
```bash
git clone <repository_url>
cd currency_detection
```

### 2. Install Required Dependencies
Ensure you have Python 3.9+ installed, then run:
```bash
pip install -r requirements.txt
```

---

## Usage Guide

### Step 1: Train the Model
Train and fine-tune the pretrained ResNet-50 model on the dataset:
```bash
python train.py
```
This script will:
* Load the dataset with multi-scale partial-crop augmentations and balanced loss weighting.
* Fine-tune `layer4` and the classification head using AdamW and Cosine Annealing.
* Save the best performing weights to `models/model.pth` and `hf_space_backend/model.pth`.

### Step 2: Run the Local Web Application
Launch the interactive Streamlit user interface:
```bash
python -m streamlit run app.py
```
Open the provided local URL (usually `http://localhost:8501`) in your web browser.

---

## Model & Training Details
* **Base Architecture:** Pretrained ResNet-50 (ImageNet weights)
* **Strategy:** Transfer Learning with deep fine-tuning of `layer4` and classification head
* **Input Resolution:** 224 x 224 pixels with aspect-ratio preserving multi-scale evaluation
* **Loss Function:** Weighted Cross-Entropy Loss (inversely proportional to class frequencies)
* **Optimizer:** AdamW (Differential learning rates: 5e-5 for `layer4`, 5e-4 for `fc`)
* **Learning Rate Schedule:** Cosine Annealing
* **Best Validation Accuracy:** 100.00%

---

## Developer Attribution
* **Developed by:** Tuhid Khan
* **GitHub Profile:** https://github.com/TuhidKhan
* **LinkedIn Profile:** https://www.linkedin.com/in/tuhid-khan/
