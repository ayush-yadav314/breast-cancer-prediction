import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# Page config (must be the first Streamlit command)
# ---------------------------------------------------------
st.set_page_config(page_title="Breast Cancer Prediction", page_icon="🩺", layout="wide")

# ---------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------
st.markdown("""
<style>
div.stButton > button {
    background-color: #ff4b4b;
    color: white;
    font-weight: bold;
    border-radius: 8px;
    height: 3em;
    width: 100%;
}
div.stButton > button:hover {
    background-color: #ff6b6b;
}
[data-testid="stMetricValue"] {
    font-size: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Dataset-wide feature averages (used to fill in the 22
# features not shown on the form)
# ---------------------------------------------------------
feature_defaults = {
    "radius_mean": 14.166077,
    "texture_mean": 19.417692,
    "perimeter_mean": 92.215868,
    "area_mean": 659.578242,
    "smoothness_mean": 0.095993,
    "compactness_mean": 0.103835,
    "concavity_mean": 0.089184,
    "concave points_mean": 0.049015,
    "symmetry_mean": 0.181497,
    "fractal_dimension_mean": 0.062715,
    "radius_se": 0.411187,
    "texture_se": 1.217882,
    "perimeter_se": 2.911569,
    "area_se": 41.279103,
    "smoothness_se": 0.006895,
    "compactness_se": 0.025323,
    "concavity_se": 0.032018,
    "concave points_se": 0.011691,
    "symmetry_se": 0.020427,
    "fractal_dimension_se": 0.003766,
    "radius_worst": 16.351514,
    "texture_worst": 25.904879,
    "perimeter_worst": 107.860484,
    "area_worst": 890.569231,
    "smoothness_worst": 0.132083,
    "compactness_worst": 0.255529,
    "concavity_worst": 0.275166,
    "concave points_worst": 0.115491,
    "symmetry_worst": 0.291364,
    "fractal_dimension_worst": 0.084124,
}

# ---------------------------------------------------------
# Load model + scaler
# ---------------------------------------------------------
model = joblib.load("models/log_reg_model.pkl")
scaler = joblib.load("models/scaler.pkl")

THRESHOLD = 0.3  # chosen earlier via recall analysis

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
col_title, col_badge = st.columns([4, 1])
with col_title:
    st.title("🩺 Breast Cancer Prediction")
    st.caption("Machine learning demo — not a medical diagnostic tool")
with col_badge:
    st.markdown("### 🔬 ML-Powered")

# ---------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------
with st.sidebar:
    st.header("Enter Measurements")
    texture_worst = st.slider("Texture Worst", min_value=12.0, max_value=50.0, value=25.9)
    radius_se = st.slider("Radius SE", min_value=0.1, max_value=3.0, value=0.41)
    concave_points_mean = st.slider("Concave Points Mean", min_value=0.0, max_value=0.2, value=0.049)
    radius_worst = st.slider("Radius Worst", min_value=8.0, max_value=36.0, value=16.35)
    area_worst = st.slider("Area Worst", min_value=180.0, max_value=4250.0, value=890.6)
    concavity_worst = st.slider("Concavity Worst", min_value=0.0, max_value=1.3, value=0.275)
    perimeter_worst = st.slider("Perimeter Worst", min_value=50.0, max_value=250.0, value=107.9)
    concave_points_worst = st.slider("Concave Points Worst", min_value=0.0, max_value=0.3, value=0.115)
    predict_clicked = st.button("Predict")

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Model Performance", "ℹ️ About"])

with tab1:
    if predict_clicked:
        input_data = feature_defaults.copy()
        input_data["texture_worst"] = texture_worst
        input_data["radius_se"] = radius_se
        input_data["concave points_mean"] = concave_points_mean
        input_data["radius_worst"] = radius_worst
        input_data["area_worst"] = area_worst
        input_data["concavity_worst"] = concavity_worst
        input_data["perimeter_worst"] = perimeter_worst
        input_data["concave points_worst"] = concave_points_worst

        input_df = pd.DataFrame([input_data])[list(feature_defaults.keys())]
        input_scaled = scaler.transform(input_df)
        probability = model.predict_proba(input_scaled)[:, 1][0]
        prediction = "Malignant" if probability >= THRESHOLD else "Benign"

        col1, col2 = st.columns(2)
        with col1:
            if prediction == "Malignant":
                st.error(f"Prediction: **{prediction}**")
            else:
                st.success(f"Prediction: **{prediction}**")
        with col2:
            st.metric("Probability of malignancy", f"{probability:.2%}")

        st.progress(float(probability))

        with st.expander("Why this prediction?"):
            st.write("Key factors considered (most influential features from SHAP analysis):")
            st.write(f"- Texture Worst: {texture_worst}")
            st.write(f"- Radius Worst: {radius_worst}")
            st.write(f"- Area Worst: {area_worst}")
            st.write(f"- Concavity Worst: {concavity_worst}")
            st.write(f"- Concave Points Worst: {concave_points_worst}")
            st.write(f"- Concave Points Mean: {concave_points_mean}")
            st.write(f"- Radius SE: {radius_se}")
            st.write(f"- Perimeter Worst: {perimeter_worst}")
            st.caption(
                "Higher values in these measurements are generally associated "
                "with malignant tumors in this model. All other features were "
                "held at their dataset-wide average."
            )
    else:
        st.info("Adjust the measurements in the sidebar and click **Predict**.")

with tab2:
    st.subheader("Model Performance")
    st.write("Model: Logistic Regression (scaled features)")
    st.write(f"Decision threshold: {THRESHOLD} (tuned to reduce missed malignant cases)")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Recall (malignant)", "98%")
    col_b.metric("Precision (malignant)", "98%")
    col_c.metric("ROC-AUC", "0.996")
    st.caption("Evaluated on a held-out 20% test split (114 samples) from the Wisconsin Breast Cancer Diagnostic dataset.")

with tab3:
    st.subheader("About this project")
    st.write(
        """
        This app classifies tumors as benign or malignant using a Logistic
        Regression model trained on cell nucleus measurements from the
        Wisconsin Breast Cancer Diagnostic dataset (569 samples, 30 features).

        Built as a portfolio project demonstrating the full machine learning
        pipeline: data cleaning, exploratory data analysis, model comparison
        (Logistic Regression, Random Forest, XGBoost), cross-validation,
        threshold tuning for recall, SHAP explainability, and deployment.

        Only the 8 most influential features (per SHAP analysis) are exposed
        here for a fast demo; the remaining 22 features are held at their
        dataset-wide average values.
        """
    )
    st.warning(
        "This tool is for educational and portfolio purposes only. It is "
        "not validated for clinical use and must not be used for real "
        "medical decisions."
    )

st.markdown("---")
st.caption("Built by Ayush | Dataset: Wisconsin Breast Cancer Diagnostic (UCI/Kaggle) | For educational purposes only")