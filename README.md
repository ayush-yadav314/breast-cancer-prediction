# 🩺 Breast Cancer Prediction

A machine learning web app that classifies breast tumors as benign or malignant based on cell nucleus measurements, using Logistic Regression trained on the Wisconsin Breast Cancer Diagnostic dataset.

**Live demo:** https://your-app-url.streamlit.app
**Repository:** https://github.com/ayush-yadav314/breast-cancer-prediction

> ⚠️ **This is an educational/portfolio project, not a medical diagnostic tool.** It is not validated for clinical use.

## Problem Statement

Classify breast tumors as benign or malignant using 30 numeric features derived from digitized images of fine needle aspirate (FNA) cell nuclei.

- **Target:** `diagnosis` (1 = malignant, 0 = benign)
- **Success metric:** recall on the malignant class (minimizing missed cancers), alongside precision and ROC-AUC

## Dataset

- **Source:** Wisconsin Breast Cancer Diagnostic dataset (Kaggle / UCI Machine Learning Repository)
- **Size:** 569 samples, 30 features (mean, standard error, and "worst" values for 10 cell nucleus characteristics: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension)
- **Class balance:** 357 benign (62.7%), 212 malignant (37.3%) — mild imbalance

## Methodology

1. **Data cleaning:** dropped the `id` column and an empty `Unnamed: 32` column; encoded `diagnosis` as 1/0
2. **Exploratory data analysis:** class distribution, feature correlation heatmap, boxplots comparing benign vs malignant distributions per feature
3. **Preprocessing:** stratified 80/20 train-test split, followed by `StandardScaler` fit on training data only (to prevent data leakage)
4. **Model training:** compared Logistic Regression, Random Forest, and XGBoost
5. **Evaluation:** confusion matrix, precision/recall/F1, ROC-AUC, and 5-fold stratified cross-validation
6. **Threshold tuning:** adjusted the decision threshold to prioritize recall (catching more true cancers) over the default 0.5 cutoff
7. **Explainability:** SHAP values to identify which features drive predictions
8. **Deployment:** saved model and scaler with `joblib`, wrapped in a Streamlit web app

## Results

### Model comparison (test set, threshold = 0.5)

| Model | Recall (malignant) | Precision (malignant) | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.93 | 0.97 | 0.996 |
| Random Forest | 0.93 | 1.00 | 0.993 |
| XGBoost | 0.90 | 1.00 | 0.990 |

### 5-fold cross-validation (recall)

All three models averaged **0.947** mean recall across folds — performance differences between models on a single split were not statistically meaningful.

### Threshold tuning (Logistic Regression)

| Threshold | False negatives | False positives | Recall |
|---|---|---|---|
| 0.5 (default) | 3 | 1 | 0.93 |
| **0.3 (chosen)** | **1** | **1** | **0.98** |
| 0.2 | 1 | 4 | 0.98 |

**Chosen model: Logistic Regression at threshold 0.3.** It reduces missed cancers from 3 to 1 without increasing false alarms, and is simpler and more interpretable than the tree-based alternatives — a relevant property for a medical-adjacent tool.

## Key SHAP Findings

- `texture_worst` was the single most influential feature for the model, more so than raw correlation with the target would suggest.
- Most features behaved intuitively (higher values → more likely malignant), consistent with the boxplot analysis.
- `compactness_se` showed a counter-intuitive direction in the model, likely due to correlation with other features — a reminder that model-level feature effects can differ from simple univariate correlation.

## Limitations

- The dataset is small (569 samples) and unusually clean; results won't generalize to noisy real-world clinical data without further validation.
- Model performance has not been validated against a clinical ground truth or reviewed by a medical professional.
- The web app exposes only 8 of 30 features for a faster demo; the rest are held at dataset-wide averages, which is a simplification, not a clinical input model.
- Slider ranges constrain inputs to realistic values found in the training data; earlier testing without these bounds showed the model produces unreliable (saturated) probabilities on out-of-range inputs.

## Tech Stack

- Python, pandas, NumPy, scikit-learn, XGBoost, SHAP
- Streamlit (web app)
- Matplotlib, Seaborn (visualization)

## Running Locally

```bash
git clone https://github.com/ayush-yadav314/breast-cancer-prediction.git
cd breast-cancer-prediction
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
streamlit run app.py
```

## Author

Ayush — B.Tech CSE (AI Specialization), Galgotias College of Engineering and Technology