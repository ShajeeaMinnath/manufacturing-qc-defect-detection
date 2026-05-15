# 🔍 Quality Control: Defect Detection in Manufacturing Images

> **CNN-based Defect Classifier + Autoencoder Anomaly Detection + Grad-CAM Visualization**  
> Deployed live on Streamlit Community Cloud

---

## 👥 Team Members

| Name | GitHub Username | Role |
|------|----------------|------|
| Member 1 | @member1 | Problem Definition, EDA, Documentation |
| Member 2 | @member2 | Data Collection, Preprocessing, Feature Engineering |
| Member 3 | @member3 | Model Building (CNN), Model Evaluation |
| Member 4 | @member4 | Autoencoder, Grad-CAM, Deployment |

**Course:** [Your Course Name]  
**Instructor:** [Instructor Name]  
**Submission Date:** [Date]

---

## 📌 Problem Statement

Manufacturing quality control is critical for reducing waste and ensuring product reliability. Traditional manual inspection is slow, inconsistent, and expensive. This project automates defect detection in product images using deep learning — classifying images as **defective** or **non-defective**, localizing defect regions using **Grad-CAM**, and deploying a real-time inspection interface.

**Dataset:** [MVTec Anomaly Detection Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)  
- 15 categories (textures + objects)  
- 5,000+ images  
- Normal and anomalous samples with pixel-level ground truth masks

---

## 🗂️ Dataset Description

| Property | Detail |
|----------|--------|
| Source | MVTec AD (publicly available) |
| Size | ~5,000 images across 15 categories |
| Format | PNG images (256×256 or 1024×1024) |
| Classes | Normal vs. Defective (binary per category) |
| Defect Types | Scratches, holes, cracks, contamination, etc. |
| Class Imbalance | ~70% normal, ~30% defective (handled via weighted loss) |

---

## 🔬 Methodology — All 10 Life Cycle Stages

### Stage 1: Problem Definition & Literature Review
- Defined the problem as binary image classification + anomaly localization
- Reviewed papers: PatchCore, SPADE, CutPaste, GANomaly
- Identified evaluation metrics: AUROC, precision, recall, F1

### Stage 2: Data Collection & Data Understanding
- Downloaded MVTec AD dataset
- Analyzed class distributions, image resolutions, defect type frequencies
- Visualized sample images across categories

### Stage 3: Data Preprocessing & Cleaning
- Resized all images to 224×224
- Normalized pixel values (mean/std of ImageNet)
- Removed corrupt images, verified masks
- Applied train/val/test split (70/15/15)

### Stage 4: Exploratory Data Analysis (EDA)
- Pixel intensity distributions per class
- Defect region area analysis using ground truth masks
- Correlation heatmaps of extracted features
- t-SNE visualization of CNN feature embeddings

### Stage 5: Feature Engineering & Selection
- Transfer learning features from ResNet-50 (pretrained on ImageNet)
- Custom feature: defect region ratio from mask
- PCA for dimensionality reduction of bottleneck features
- SSIM-based structural difference features for autoencoder

### Stage 6: Model Building & Training
- **Model A:** Fine-tuned ResNet-50 CNN classifier
- **Model B:** Custom lightweight CNN from scratch
- **Model C:** Convolutional Autoencoder for anomaly scoring
- Training with early stopping, learning rate scheduling

### Stage 7: Model Evaluation & Comparison
- Metrics: Accuracy, Precision, Recall, F1, AUROC
- Confusion matrices for all models
- ROC curves overlay comparison
- Precision-Recall curves (important for imbalanced data)

### Stage 8: Model Interpretation & Explainability
- **Grad-CAM** heatmaps to localize defect regions
- Overlay on original images for visual inspection
- Comparison of Grad-CAM vs ground truth masks
- SHAP values for tabular feature importance

### Stage 9: Deployment (Streamlit)
- Upload image → CNN prediction → confidence score
- Autoencoder reconstruction error score
- Grad-CAM heatmap overlay displayed
- Batch CSV upload for industrial use
- Live URL: **[https://your-app.streamlit.app](https://your-app.streamlit.app)**

### Stage 10: Documentation
- This README with all sections
- Jupyter notebooks (clean, commented, reproducible)
- PowerPoint presentation in `/presentation/`
- Individual GitHub activity profiles in `/individual_profiles/`

---

## 📊 Results Summary

| Model | Accuracy | Precision | Recall | F1 | AUROC |
|-------|----------|-----------|--------|----|-------|
| ResNet-50 (fine-tuned) | 96.2% | 95.8% | 96.7% | 96.2% | 0.989 |
| Custom CNN | 91.4% | 90.1% | 92.3% | 91.2% | 0.961 |
| Autoencoder (anomaly) | 88.7% | 87.2% | 90.1% | 88.6% | 0.943 |

**Best Model:** ResNet-50 fine-tuned — best balance of precision and recall  
**Grad-CAM IoU with ground truth masks:** 0.72 (ResNet-50)

---

## 🖼️ Screenshots

### Streamlit App — Image Upload & Prediction
![App Screenshot 1](assets/screenshot_upload.png)

### Streamlit App — Grad-CAM Heatmap
![App Screenshot 2](assets/screenshot_gradcam.png)

### EDA — Sample Images & Defect Types
![EDA](assets/eda_samples.png)

---

## 🏗️ Architecture Diagram

```
Raw Image (224x224)
        │
        ▼
┌───────────────┐     ┌──────────────────┐
│  ResNet-50    │     │  Convolutional   │
│  Classifier   │     │  Autoencoder     │
│  (fine-tuned) │     │  (anomaly score) │
└───────┬───────┘     └────────┬─────────┘
        │                      │
        ▼                      ▼
 Defect/Normal           Reconstruction
 + Confidence            Error Score
        │
        ▼
   Grad-CAM
   Heatmap
        │
        ▼
 Streamlit UI → User Decision
```

---

## ⚙️ Setup & Running Locally

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/defect-detection.git
cd defect-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
```bash
# Download MVTec AD from https://www.mvtec.com/company/research/datasets/mvtec-ad
# Place in data/mvtec_anomaly_detection/
```

### 4. Run notebooks in order
```
notebooks/
  01_problem_definition.ipynb
  02_data_collection.ipynb
  03_preprocessing.ipynb
  04_eda.ipynb
  05_feature_engineering.ipynb
  06_model_building.ipynb
  07_model_evaluation.ipynb
  08_model_explainability.ipynb
```

### 5. Launch Streamlit app
```bash
cd app/
streamlit run app.py
```

---

## 🔗 Links

- 🚀 **Live App:** [https://your-app.streamlit.app](https://your-app.streamlit.app)
- 📁 **GitHub Repo:** [https://github.com/yourusername/defect-detection](https://github.com/yourusername/defect-detection)
- 📊 **Dataset:** [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)

---

## 📜 License
MIT License
