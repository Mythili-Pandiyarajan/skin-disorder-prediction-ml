# 🩺 Skin Disorder Prediction ML Project
An end-to-end Machine Learning project to predict erythemato-squamous skin diseases using clinical and histopathological features across 6 disease classes.

## 🚀 Live Demo
🤗 **HuggingFace:** https://huggingface.co/spaces/Mythili-Pandiyarajan/SkinDisorderPredictor

## 🔍 Project Overview
Erythemato-squamous diseases are notoriously difficult to diagnose because they all share the clinical features of erythema (redness) and scaling — with subtle differences visible only through biopsy. Misdiagnosis leads to incorrect treatment and prolonged patient suffering.

This project builds a multi-class classification system to predict the correct skin disorder from clinical and histopathological features.

**6 Disease Classes:**
- Psoriasis
- Seborrheic Dermatitis
- Lichen Planus
- Pityriasis Rosea
- Chronic Dermatitis
- Pityriasis Rubra Pilaris

**Models Compared:**
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- XGBoost Classifier
- 1D CNN (Deep Learning — Bonus)
- 2D CNN Heatmap (Computer Vision Angle — Bonus)

## 📁 Project Structure

| File | Description |
|------|-------------|
| `skin_disorder_prediction.ipynb` | ML notebook with EDA, models, SHAP, CNN |
| `app.py` | Gradio / Streamlit web application |
| `skin_disorder_model.pkl` | Best model — Random Forest Tuned |
| `skin_disorder_scaler.pkl` | StandardScaler for inference |
| `requirements.txt` | Required libraries |

## 🛠️ Libraries Used
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- Streamlit / Gradio
- Matplotlib, Seaborn
- SHAP

## 🚀 How to Run Locally
```bash
git clone https://github.com/Mythili-Pandiyarajan/skin-disorder-prediction.git
cd skin-disorder-prediction
pip install -r requirements.txt
streamlit run app.py
```

## 📊 Model Performance

| Model | Accuracy | F1 Macro |
|-------|----------|----------|
| **Random Forest (Tuned) ✅** | **0.9730** | **0.9694** |
| Logistic Regression (Baseline) | 0.9595 | 0.9574 |
| Random Forest (Baseline) | 0.9595 | 0.9545 |
| Decision Tree (Baseline) | 0.9595 | 0.9447 |
| XGBoost (Baseline) | 0.9324 | 0.9257 |
| XGBoost (Tuned) | 0.9324 | 0.9257 |
| Decision Tree (Tuned) | 0.9324 | 0.9245 |
| 1D CNN (Deep Learning) | 0.9324 | 0.8837 |
| 2D CNN Heatmap (CV Angle) | 0.3108 | 0.07xx |

✅ **Best Model: Random Forest Tuned** with Accuracy **97.30%** and F1 Macro **0.9694**
- 4 out of 6 disease classes achieve **perfect recall (1.00)**: Psoriasis, Lichen Planus, Pityriasis Rosea, Pityriasis Rubra Pilaris

## 📌 Dataset
- **366 patient records**, 34 features
- **Clinical features** (0–3 ordinal scale): erythema, scaling, itching, koebner phenomenon, polygonal papules, and more
- **Histopathological features** (0–3 ordinal scale): melanin incontinence, acanthosis, parakeratosis, band-like infiltrate, and more
- **Age** (continuous) — missing values encoded as `?`, imputed with median
- Source: UCI Dermatology Dataset (Dermatology Database)

## 🔬 Key Features (Top SHAP)
| Feature | Type | Importance |
|---------|------|------------|
| `melanin_incontinence` | Histopathological | Highest |
| `acanthosis` | Histopathological | High |
| `parakeratosis` | Histopathological | High |
| `band-like_infiltrate` | Histopathological | High |
| `koebner_phenomenon` | Clinical | Medium |
| `polygonal_papules` | Clinical | Medium |
| `Age` | Demographic | Medium |

## 📓 Notebook Structure
1. Import Libraries
2. Load & Understand Data
3. Basic Checks
4. Exploratory Data Analysis (Univariate, Bivariate, Multivariate)
5. Data Preprocessing (Missing values, Outliers)
6. Model Building & Evaluation (4 baseline models)
7. Hyperparameter Tuning (GridSearchCV)
8. Cross Validation
9. Overfitting Check
10. Confusion Matrix
11. Feature Importance
12. SHAP Explainability
13. Deep Learning — 1D CNN on Tabular Features *(Bonus)*
14. Computer Vision Angle — Tabular → Heatmap → 2D CNN *(Bonus)*
15. Model Comparison Report
16. Clinical Suggestions to Dermatologists
17. Challenges Faced & Solutions
18. Conclusion & Model Saving

## ⚙️ Tech Stack
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189AB4?style=flat)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)

- **ML:** Random Forest, XGBoost, Decision Tree, Logistic Regression
- **Deep Learning:** 1D CNN, 2D CNN (TensorFlow/Keras)
- **Explainability:** SHAP TreeExplainer
- **UI:** Gradio (HuggingFace Spaces)

## 💡 Key Findings
- **Histopathological features dominate** — biopsy-derived features are far more discriminating than clinical symptoms alone
- **Random Forest Tuned beats all models** including XGBoost and Deep Learning — 366 rows is too small for CNN to generalize
- **1D CNN (93.24%)** is competitive but can't beat RF Tuned (97.30%) on this dataset size
- **2D CNN Heatmap** approach did not generalize — converting tabular data to heatmap loses too much information on small datasets
- **Koebner phenomenon** and **polygonal papules** are sparse but highly disease-specific when present

## 🩺 Clinical Recommendations
1. **Always order a biopsy** when clinical symptoms overlap — histopathological features resolve ambiguity clinical exam alone cannot
2. **Document Koebner phenomenon** carefully — it strongly differentiates Psoriasis from other classes
3. **Combine clinical + histopathological features** for best diagnostic accuracy — neither alone is sufficient

## 👩‍💻 Author
**Mythili Pandiyarajan** — [GitHub Profile](https://github.com/Mythili-Pandiyarajan)
