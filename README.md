# Annual Insurance Premium Prediction

An end-to-end machine learning project predicting annual insurance premiums from customer demographic, health, and financial data — covering the full ML lifecycle from raw data to a deployed Streamlit app.

**Live App:** [Add your Streamlit Cloud link here after deployment]

## Overview

This project predicts `Annual_Premium_Amount` for 50,000 insurance customers using an XGBoost regression model, achieving a **Test R² of 0.9806** and a median prediction error of **₹398**.

## Project Structure

```
premium-prediction/
├── app.py                              # Streamlit web app
├── requirements.txt                    # Python dependencies
├── models/
│   ├── premium_xgb_model_23feat.pkl    # Trained XGBoost model
│   ├── premium_feature_list_23feat.pkl # Feature column order
│   └── premium_encoders_23feat.pkl     # LabelEncoders for categorical fields
├── notebooks/
│   └── premiums.ipynb                  # Full EDA + modeling notebook
└── README.md
```

## ML Lifecycle

**1. Data Understanding / EDA**
- 50,000 rows, 13 raw columns
- Univariate and bivariate analysis on all numeric and categorical features
- Correlation heatmap (numeric features) and Cramér's V (categorical associations)
- Identified confounding relationship: `Marital_status` was a proxy for `Age`

**2. Data Cleaning**
- Dropped 13 rows with random, patternless nulls (`Employment_Status`, `Income_Level`)
- Standardized 4 inconsistent `Smoking_Status` category spellings into one
- Dropped 58 rows with corrupted `Age` values (e.g., 356 instead of a valid age)
- Fixed sign errors in `Number Of Dependants` (negative values via `abs()`)

**3. Feature Engineering**
- Split compound `Medical History` strings into 4 binary disease flags + a disease count
- Created life-stage `Age` bands based on premium step-changes observed in EDA
- Tested multiple interaction/ratio features (BMI × Smoking, Age × Disease Count, Income/Dependant ratio, Gender × Smoking) — kept only those that measurably improved signal

**4. Feature Selection**
- **VIF** — removed perfectly redundant features (`disease_count` vs. its 4 components)
- **Mutual Information** — confirmed `Region` carries zero predictive signal (MI = 0.000)
- **RFE** (Random Forest estimator) — final feature ranking and selection

**5. Model Selection**
| Model | CV R² |
|---|---|
| Linear Regression | 0.719 |
| Random Forest | 0.9465 |
| **XGBoost (final)** | **0.9806** |

- 5-fold cross-validation used throughout, with a held-out test set never touched until final evaluation
- Train vs. test R² gap kept under 1 point across all experiments — confirmed no overfitting

**6. Final Result**
- **Test R²: 0.9806**
- **Test MAE: ₹793** | **Median Error: ₹398**
- 70.1% of predictions within 10% of actual premium

## Running Locally

```bash
git clone https://github.com/YOUR_USERNAME/premium-prediction.git
cd premium-prediction
pip install -r requirements.txt
streamlit run app.py
```

## Tech Stack

- **Python** — pandas, numpy, scikit-learn, XGBoost
- **Streamlit** — web app / deployment
- **Model** — XGBoost Regressor (23 engineered features)

## Author

[Your Name] — [LinkedIn] | [Portfolio]
