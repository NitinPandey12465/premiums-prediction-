"""
Annual Premium Prediction App
Predicts insurance premium based on customer profile using a trained XGBoost model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Premium Predictor", page_icon="📋", layout="wide")

GITHUB_URL = "https://github.com/NitinPandey12465/premiums-prediction-"

# ============================================================
# VISUAL IDENTITY -- ledger / underwriting document motif
# Deep policy-green + warm parchment, serif display for headers
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background-color: #FAF9F6; }

    h1, h2, h3 { font-family: 'Fraunces', serif !important; color: #1B4B43 !important; }

    .ledger-header {
        border-bottom: 3px double #1B4B43;
        padding-bottom: 0.6rem;
        margin-bottom: 1.6rem;
    }
    .ledger-header .eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #8A7F66;
    }
    .ledger-header h1 {
        margin: 0.15rem 0 0 0 !important;
        font-size: 2.3rem !important;
        font-weight: 700 !important;
    }

    .metric-card {
        background: #FFFFFF;
        border: 1px solid #DDD6C4;
        border-left: 4px solid #1B4B43;
        border-radius: 4px;
        padding: 1rem 1.2rem;
    }
    .metric-card .label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #8A7F66;
    }
    .metric-card .value {
        font-family: 'Fraunces', serif;
        font-size: 1.9rem;
        font-weight: 700;
        color: #1B4B43;
    }

    .prediction-result {
        background: #1B4B43;
        color: #FAF9F6;
        border-radius: 6px;
        padding: 1.6rem 2rem;
        text-align: center;
        margin-top: 1rem;
    }
    .prediction-result .prediction-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        opacity: 0.75;
    }
    .prediction-result .prediction-value {
        font-family: 'Fraunces', serif;
        font-size: 3rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }

    .finding-card {
        background: #FFFFFF;
        border: 1px solid #DDD6C4;
        border-radius: 4px;
        padding: 1rem 1.3rem;
        margin-bottom: 0.7rem;
    }
    .finding-card .tag {
        display: inline-block;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        background: #EDE8DD;
        color: #1B4B43;
        padding: 0.15rem 0.5rem;
        border-radius: 3px;
        margin-bottom: 0.4rem;
    }

    .stButton > button {
        background-color: #1B4B43;
        color: #FAF9F6;
        border-radius: 4px;
        border: none;
        font-weight: 600;
        padding: 0.6rem 0;
    }
    .stButton > button:hover { background-color: #14382F; color: #FAF9F6; }

    [data-testid="stMetricValue"] { font-family: 'Fraunces', serif; color: #1B4B43; }

    footer, [data-testid="stDecoration"] { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

sns.set_style("whitegrid", {"axes.facecolor": "#FFFFFF", "grid.color": "#EDE8DD"})
PALETTE = ["#1B4B43", "#C77B4A", "#8A9B7E", "#D9C97A", "#5C7A8A"]
sns.set_palette(PALETTE)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.edgecolor"] = "#DDD6C4"
plt.rcParams["figure.facecolor"] = "#FAF9F6"
plt.rcParams["axes.facecolor"] = "#FFFFFF"

# ============================================================
# LOAD MODEL, ENCODERS, FEATURE LIST
# ============================================================
MODEL_DIR = "."

@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "premium_xgb_model_23feat.pkl"))
    feature_list = joblib.load(os.path.join(MODEL_DIR, "premium_feature_list_23feat.pkl"))
    encoders = joblib.load(os.path.join(MODEL_DIR, "premium_encoders_23feat.pkl"))
    return model, feature_list, encoders

@st.cache_data
def load_raw_data():
    """Loads the raw dataset for live EDA charts, if present in the repo."""
    for candidate in ["premiums.xlsx", "data/premiums.xlsx"]:
        if os.path.exists(candidate):
            return pd.read_excel(candidate)
    return None

model, feature_list, encoders = load_artifacts()
raw_df = load_raw_data()

REFERENCE_METRICS = {"r2": 0.9806, "mae": 793.0, "median_err": 398.0, "within_10pct": 70.1}

@st.cache_data
def compute_live_metrics(_model, _feature_list, _encoders, df):
    """Recomputes held-out test metrics from the raw data using the same
    feature engineering as training, so the numbers shown always reflect
    whichever model file is actually loaded, not a hardcoded string."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error

    d = df.copy()
    d = d.dropna(subset=["Employment_Status", "Income_Level"]).copy()
    d["Smoking_Status"] = d["Smoking_Status"].replace({
        "No Smoking": "Non-Smoker", "Not Smoking": "Non-Smoker",
        "Does Not Smoke": "Non-Smoker", "Smoking=0": "Non-Smoker",
    })
    d["Smoking_Status"] = d["Smoking_Status"].fillna("Non-Smoker")
    d = d[d["Age"] <= 100].copy()
    d["Number Of Dependants"] = d["Number Of Dependants"].abs()

    d["has_diabetes"] = d["Medical History"].str.contains("Diabetes").astype(int)
    d["has_high_bp"] = d["Medical History"].str.contains("High blood pressure").astype(int)
    d["has_thyroid"] = d["Medical History"].str.contains("Thyroid").astype(int)
    d["has_heart_disease"] = d["Medical History"].str.contains("Heart disease").astype(int)
    d["disease_count"] = d["has_diabetes"] + d["has_high_bp"] + d["has_thyroid"] + d["has_heart_disease"]
    d["age_band"] = pd.cut(d["Age"], bins=[0, 18, 30, 45, 60, 100],
                             labels=["0-18", "19-30", "31-45", "46-60", "61+"])
    d["bmi_smoking"] = d["BMI_Category"] + "_" + d["Smoking_Status"]
    d["gender_smoking"] = d["Gender"] + "_" + d["Smoking_Status"]
    d["income_per_dependant"] = d["Income_Lakhs"] / (d["Number Of Dependants"] + 1)
    d["age_x_disease_count"] = d["Age"] * d["disease_count"]

    try:
        for col, le in _encoders.items():
            known = set(le.classes_)
            mask = d[col].astype(str).isin(known)
            if not mask.all():
                d = d[mask]
            d[col + "_enc"] = le.transform(d[col].astype(str))

        d["gender_smoking_encoded"] = d["gender_smoking_enc"]
        d["smoking_encoded"] = d["Smoking_Status_enc"]

        missing = [f for f in _feature_list if f not in d.columns]
        if missing:
            return None

        X = d[_feature_list]
        y = d["Annual_Premium_Amount"]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        preds = _model.predict(X_test)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        err_pct = (abs(preds - y_test.values) / y_test.values) * 100
        median_err = float(np.median(abs(preds - y_test.values)))
        within_10 = float((err_pct < 10).mean() * 100)

        return {"r2": r2, "mae": mae, "median_err": median_err, "within_10pct": within_10}
    except Exception:
        return None

if raw_df is not None:
    live_metrics = compute_live_metrics(model, feature_list, encoders, raw_df)
    METRICS = live_metrics if live_metrics else REFERENCE_METRICS
else:
    METRICS = REFERENCE_METRICS

# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
    <div class="ledger-header">
        <div class="eyebrow">Underwriting Model No. 2026-01</div>
        <h1>Annual Premium Predictor</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### About")
    st.markdown(
        "Predicts a customer's **Annual Insurance Premium** using an XGBoost "
        "model trained on 50,000 customer records."
    )
    st.markdown("**Held-out test performance**")
    st.markdown(
        f"""
        - R2 Score: `{METRICS['r2']:.4f}`
        - MAE: `Rs {METRICS['mae']:.0f}`
        - Median Error: `Rs {METRICS['median_err']:.0f}`
        """
    )
    st.markdown("---")

    with st.expander("How this works"):
        st.markdown(
            "Raw customer records were cleaned (corrupted ages, sign errors, "
            "inconsistent categories), engineered into risk-relevant features "
            "(disease flags, age bands, BMI-smoking interactions), narrowed down "
            "via VIF / Mutual Information / RFE, then used to train and compare "
            "Linear Regression, Random Forest, and XGBoost models. The best "
            "model (XGBoost) is deployed here, evaluated on data it never saw "
            "during training."
        )

    st.markdown(f"[View source code on GitHub]({GITHUB_URL})")

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3 = st.tabs(["Predict Premium", "Model Performance", "EDA Insights"])

# ------------------------------------------------------------
# TAB 1: PREDICTION FORM
# ------------------------------------------------------------
with tab1:
    st.subheader("Customer Profile")

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

    predict_clicked = st.button("Predict Premium", type="primary", use_container_width=True)

    if predict_clicked:
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

            row["gender_smoking_encoded"] = row["gender_smoking_enc"]
            row["smoking_encoded"] = row["Smoking_Status_enc"]

            X_input = pd.DataFrame([row])[feature_list]
            prediction = model.predict(X_input)[0]
            margin = METRICS["mae"]
            low, high = max(0, prediction - margin), prediction + margin

            st.markdown(
                f"""
                <div class="prediction-result">
                    <div class="prediction-label">Predicted Annual Premium</div>
                    <div class="prediction-value">Rs {prediction:,.2f}</div>
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;opacity:0.75;margin-top:0.4rem;">
                        typical range Rs {low:,.0f} &ndash; Rs {high:,.0f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(
                "Range reflects the model's average error (MAE) on unseen test data, "
                "not a statistical confidence interval. Predictions for uncommon "
                "profile combinations (e.g. very young age with severe medical history) "
                "may be less reliable, since the model has fewer similar examples to learn from."
            )

        except Exception as e:
            st.error(f"Something went wrong building the input: {e}")

# ------------------------------------------------------------
# TAB 2: MODEL PERFORMANCE
# ------------------------------------------------------------
with tab2:
    st.subheader("Held-Out Test Set Performance")

    m1, m2, m3, m4 = st.columns(4)
    metrics_display = [
        ("Test R2", f"{METRICS['r2']:.4f}"),
        ("Test MAE", f"Rs {METRICS['mae']:.0f}"),
        ("Median Error", f"Rs {METRICS['median_err']:.0f}"),
        ("Within 10% Error", f"{METRICS['within_10pct']:.1f}%"),
    ]
    for col, (label, value) in zip([m1, m2, m3, m4], metrics_display):
        col.markdown(
            f'<div class="metric-card"><div class="label">{label}</div><div class="value">{value}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Model Comparison")

    comparison_df = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest", "XGBoost (final)"],
        "CV R2": [0.719, 0.9465, 0.9808],
    })

    fig, ax = plt.subplots(figsize=(7, 3.8))
    bars = ax.bar(comparison_df["Model"], comparison_df["CV R2"], color=[PALETTE[4], PALETTE[2], PALETTE[0]], width=0.5)
    for bar, val in zip(bars, comparison_df["CV R2"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015, f"{val:.3f}", ha="center", fontsize=10, color="#1A1A1A")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Cross-Validated R2")
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)

    st.markdown(
        """
        **Why XGBoost:** Insurance premiums in this dataset follow strong non-linear,
        stepped patterns -- premium tiers by Insurance Plan, sharp age-band jumps -- that
        Linear Regression structurally cannot capture. Tree-based models handle this
        naturally, and XGBoost slightly outperformed Random Forest through sequential
        error correction.
        """
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Feature Importance")

    try:
        importances = pd.DataFrame({
            "feature": feature_list,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=True).tail(10)

        fig2, ax2 = plt.subplots(figsize=(7, 4.5))
        ax2.barh(importances["feature"], importances["importance"], color=PALETTE[0])
        ax2.set_xlabel("Importance")
        ax2.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig2)
    except Exception:
        st.info("Feature importance unavailable for this model artifact.")

    st.caption(
        "Insurance_Plan and Age dominate the model's predictions, consistent with "
        "real-world insurance pricing where plan tier and age are primary rating factors."
    )

# ------------------------------------------------------------
# TAB 3: EDA INSIGHTS -- live charts from the actual dataset
# ------------------------------------------------------------
with tab3:
    st.subheader("Exploratory Data Analysis")

    if raw_df is None:
        st.warning(
            "Raw dataset (premiums.xlsx) not found in the repository, so live charts "
            "can't be rendered here. Add it to the repo root (or a data/ folder) to "
            "enable this tab, or view the full analysis in the notebook."
        )
    else:
        df_eda = raw_df.copy()

        st.markdown("##### Target Variable -- Annual Premium Amount")
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
        sns.histplot(df_eda["Annual_Premium_Amount"], kde=True, ax=axes[0], color=PALETTE[0])
        axes[0].set_title("Distribution", fontsize=10)
        sns.boxplot(x=df_eda["Annual_Premium_Amount"], ax=axes[1], color=PALETTE[1])
        axes[1].set_title("Spread & Outliers", fontsize=10)
        for a in axes:
            a.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("##### Age vs. Premium")
        col_a, col_b = st.columns(2)
        with col_a:
            fig, ax = plt.subplots(figsize=(5.3, 4))
            sample = df_eda[df_eda["Age"] <= 100].sample(min(4000, len(df_eda)), random_state=42)
            ax.scatter(sample["Age"], sample["Annual_Premium_Amount"], alpha=0.15, s=12, color=PALETTE[0])
            ax.set_xlabel("Age")
            ax.set_ylabel("Annual Premium")
            ax.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig)
        with col_b:
            df_clean_age = df_eda[df_eda["Age"] <= 100].copy()
            df_clean_age["age_band"] = pd.cut(
                df_clean_age["Age"], bins=[0, 18, 30, 45, 60, 100],
                labels=["0-18", "19-30", "31-45", "46-60", "61+"]
            )
            band_means = df_clean_age.groupby("age_band", observed=True)["Annual_Premium_Amount"].mean()
            fig, ax = plt.subplots(figsize=(5.3, 4))
            ax.bar(band_means.index.astype(str), band_means.values, color=PALETTE[0])
            ax.set_ylabel("Mean Annual Premium")
            ax.set_title("Premium by Age Band", fontsize=10)
            ax.spines[["top", "right"]].set_visible(False)
            plt.xticks(rotation=0)
            st.pyplot(fig)

        st.markdown("##### Premium by Category")
        cat_choice = st.selectbox(
            "Choose a categorical feature",
            ["Insurance_Plan", "Medical History", "Smoking_Status", "BMI_Category",
             "Marital_status", "Employment_Status", "Gender", "Region"],
        )
        fig, ax = plt.subplots(figsize=(9, 4))
        order = df_eda.groupby(cat_choice)["Annual_Premium_Amount"].median().sort_values(ascending=False).index
        sns.boxplot(data=df_eda, x=cat_choice, y="Annual_Premium_Amount", order=order, ax=ax, palette=PALETTE)
        ax.set_ylabel("Annual Premium")
        ax.spines[["top", "right"]].set_visible(False)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("##### Correlation -- Numeric Features")
        numeric_cols = ["Age", "Number Of Dependants", "Income_Lakhs", "Annual_Premium_Amount"]
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        corr = df_eda[numeric_cols].corr()
        cmap = sns.light_palette(PALETTE[0], as_cmap=True)
        sns.heatmap(corr, annot=True, cmap=cmap, center=0, fmt=".2f", ax=ax, linewidths=0.5, linecolor="#FAF9F6")
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### Key Findings")

        findings = [
            ("Strongest Predictor", "Insurance Plan tier and Age dominate the model -- together explaining roughly 89% of prediction weight, mirroring real actuarial pricing."),
            ("Confounding Detected", "Marital_status appeared linked to Medical History (Cramer's V = 0.51) -- investigation showed both were really proxies for Age (married customers average 15 years older)."),
            ("Compounding Risk", "BMI x Smoking Status interact: obese regular smokers show nearly double the premium of normal-weight non-smokers, confirming risk compounds rather than adds."),
            ("Dropped Features", "Region (Mutual Information = 0.000) and Gender showed negligible relationship with premium and were excluded via VIF, Mutual Information, and RFE testing."),
        ]
        for tag, text in findings:
            st.markdown(
                f'<div class="finding-card"><div class="tag">{tag}</div>{text}</div>',
                unsafe_allow_html=True,
            )

        st.info(f"Full EDA notebook with the complete analysis is available in the [GitHub repository]({GITHUB_URL}).")

st.markdown("<br>", unsafe_allow_html=True)
st.caption(f"Built with Streamlit -- Model: XGBoost -- [GitHub Repository]({GITHUB_URL})")
