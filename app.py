"""
Annual Premium Prediction App
Predicts insurance premium based on customer profile using a trained XGBoost model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Premium Predictor", page_icon="💰", layout="wide")

# ============================================================
# LOAD MODEL, ENCODERS, FEATURE LIST
# ============================================================
MODEL_DIR = "models"

@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "premium_xgb_model_23feat.pkl"))
    feature_list = joblib.load(os.path.join(MODEL_DIR, "premium_feature_list_23feat.pkl"))
    encoders = joblib.load(os.path.join(MODEL_DIR, "premium_encoders_23feat.pkl"))
    return model, feature_list, encoders

model, feature_list, encoders = load_artifacts()

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("Annual Premium Predictor")
st.sidebar.markdown(
    """
    Predicts a customer's **Annual Insurance Premium** using an XGBoost model
    trained on 50,000 customer records.

    **Model performance (held-out test set):**
    - R² Score: **0.9806**
    - MAE: **₹793**
    - Median Error: **₹398**
    """
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "🔗 [View source code on GitHub](https://github.com/YOUR_USERNAME/premium-prediction)"
)

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3 = st.tabs(["🔮 Predict Premium", "📊 Model Performance", "🔍 EDA Insights"])

# ------------------------------------------------------------
# TAB 1: PREDICTION FORM
# ------------------------------------------------------------
with tab1:
    st.header("Enter Customer Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 18, 80, 35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        region = st.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])
        marital_status = st.selectbox("Marital Status", ["Married", "Unmarried"])

    with col2:
        dependants = st.slider("Number Of Dependants", 0, 5, 1)
        bmi_category = st.selectbox("BMI Category", ["Normal", "Overweight", "Obesity", "Underweight"])
        smoking_status = st.selectbox("Smoking Status", ["Non-Smoker", "Occasional", "Regular"])
        employment_status = st.selectbox("Employment Status", ["Salaried", "Self-Employed", "Freelancer"])

    with col3:
        income_lakhs = st.number_input("Income (Lakhs)", min_value=1, max_value=930, value=10)
        insurance_plan = st.selectbox("Insurance Plan", ["Bronze", "Silver", "Gold"])
        medical_history = st.selectbox(
            "Medical History",
            ["No Disease", "Diabetes", "High blood pressure", "Thyroid", "Heart disease",
             "Diabetes & High blood pressure", "Diabetes & Thyroid", "Diabetes & Heart disease",
             "High blood pressure & Heart disease"]
        )

    if st.button("Predict Premium", type="primary", use_container_width=True):
        # ---- Feature engineering, matching the training pipeline exactly ----
        has_diabetes = int("Diabetes" in medical_history)
        has_high_bp = int("High blood pressure" in medical_history)
        has_thyroid = int("Thyroid" in medical_history)
        has_heart_disease = int("Heart disease" in medical_history)
        disease_count = has_diabetes + has_high_bp + has_thyroid + has_heart_disease

        if age <= 18:
            age_band = "0-18"
        elif age <= 30:
            age_band = "19-30"
        elif age <= 45:
            age_band = "31-45"
        elif age <= 60:
            age_band = "46-60"
        else:
            age_band = "61+"

        bmi_smoking = f"{bmi_category}_{smoking_status}"
        gender_smoking = f"{gender}_{smoking_status}"
        income_per_dependant = income_lakhs / (dependants + 1)
        age_x_disease_count = age * disease_count

        row = {
            "Age": age,
            "Number Of Dependants": dependants,
            "Income_Lakhs": income_lakhs,
            "has_diabetes": has_diabetes,
            "has_high_bp": has_high_bp,
            "has_thyroid": has_thyroid,
            "has_heart_disease": has_heart_disease,
            "disease_count": disease_count,
            "income_per_dependant": income_per_dependant,
            "age_x_disease_count": age_x_disease_count,
        }

        # Encode categorical fields using the saved encoders
        cat_values = {
            "Gender": gender,
            "Region": region,
            "Marital_status": marital_status,
            "BMI_Category": bmi_category,
            "Smoking_Status": smoking_status,
            "Employment_Status": employment_status,
            "Medical History": medical_history,
            "Insurance_Plan": insurance_plan,
            "age_band": age_band,
            "bmi_smoking": bmi_smoking,
            "gender_smoking": gender_smoking,
        }

        try:
            for col, val in cat_values.items():
                le = encoders[col]
                if val in le.classes_:
                    row[col + "_enc"] = int(le.transform([val])[0])
                else:
                    st.error(f"Value '{val}' for {col} was not seen during training.")
                    st.stop()

            # gender_smoking_encoded / smoking_encoded — extra columns from earlier testing
            row["gender_smoking_encoded"] = row["gender_smoking_enc"]
            row["smoking_encoded"] = row["Smoking_Status_enc"]

            X_input = pd.DataFrame([row])[feature_list]
            prediction = model.predict(X_input)[0]

            st.success(f"### Predicted Annual Premium: ₹{prediction:,.2f}")

        except Exception as e:
            st.error(f"Something went wrong building the input: {e}")

# ------------------------------------------------------------
# TAB 2: MODEL PERFORMANCE
# ------------------------------------------------------------
with tab2:
    st.header("Model Performance")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test R²", "0.9806")
    m2.metric("Test MAE", "₹793")
    m3.metric("Median Error", "₹398")
    m4.metric("Within 10% Error", "70.1%")

    st.markdown("---")

    st.subheader("Model Comparison")
    comparison_df = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest", "XGBoost (final)"],
        "CV R²": [0.719, 0.9465, 0.9808],
    })
    st.bar_chart(comparison_df.set_index("Model"))

    st.markdown(
        """
        **Why XGBoost:** Insurance premiums in this dataset follow strong non-linear,
        stepped patterns (e.g., premium tiers by Insurance Plan, age-band jumps) that
        Linear Regression cannot capture. Tree-based models handle this naturally,
        and XGBoost slightly outperformed Random Forest through sequential error correction.
        """
    )

    st.subheader("Feature Importance")
    st.markdown(
        "`Insurance_Plan` and `Age` dominate the model's predictions, consistent with "
        "real-world insurance pricing where plan tier and age are primary rating factors."
    )

# ------------------------------------------------------------
# TAB 3: EDA INSIGHTS
# ------------------------------------------------------------
with tab3:
    st.header("Key EDA Insights")

    st.markdown(
        """
        - **Insurance Plan** is the strongest predictor — Gold plans average significantly
          higher premiums than Bronze, reflecting tiered pricing.
        - **Age** shows a strong positive relationship with premium (correlation ≈ 0.77
          after cleaning), consistent with real insurance risk pricing.
        - **Medical History** (particularly Heart Disease and Diabetes) meaningfully
          raises predicted premiums, matching real actuarial risk factors.
        - **BMI × Smoking Status** compounds — obese regular smokers show nearly double
          the premium of normal-weight non-smokers.
        - **Region** and **Gender** showed negligible relationship with premium and were
          excluded from the final feature set based on Mutual Information and
          Cramér's V testing.
        """
    )

    st.info(
        "Full EDA notebook with all charts (univariate, bivariate, correlation heatmap, "
        "Cramér's V) is available in the GitHub repository under `/notebooks`."
    )

st.markdown("---")
st.caption("Built with Streamlit • Model: XGBoost • [GitHub Repository](https://github.com/YOUR_USERNAME/premium-prediction)")
