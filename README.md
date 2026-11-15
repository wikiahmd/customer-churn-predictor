# 📞 Telco Retention Lab - Customer Churn Predictor

**Live Streamlit app:**

**GitHub Link:** 

**LinkedIn:** 

A portfolio project from the PSEB Artificial Intelligence Bootcamp brief. Explore historical telecom customer churn, predict churn risk for a hypothetical customer and recommend retention experiments.

**Not a live business model.** This is a tutorial trained on a public historical dataset. Risk scores are not calibrated probabilities and group differences are not proven cause of churn.

## Business question

Which customer attributes are associated with churn, how well can a classifier
catch churners, and how might a retention team use its risk scores responsibly?

## Data and cleaning

- 7,043 customers, 21 original columns; churn rate **26.54%** (1, 869 churned).
- `TotalCharges` was text with **11 blank values**. All 11 had `tenure=0`, so they 
were set to `0.0`.
- No duplicate rows or customer IDs were found. `customerID` was excluded from
predictors.
- The target was encoded as `Churn=Yes --> 1` and `Churn=No --> 0`.

## Key exploratory findiings

| Segment | Historical churn rate |
|---|---:|
| Month-to-mon contract | 42.7% |
| Two-year contract | 2.8% |
| Electronic check | 45.3% |
| Automatic credit-card payment | 15.2% |
| 0-12 months' tenure | 47.4% |
| 49-72 months' tenure | 9.5% |

These are associations, not causal claims. Customers with higher monthly bills
also tend to have higher churn than the lowest charge quartile, but the pattern is
not perfectly increasing.

## Models and validation

- **80/20 stratified** train/test split (`random_state=42`).
- Preprocessing in an `sklearn` `Pipeline` and `ColumnTransformer`: scale
numerical inputs; one-hot encode categoricla inputs; handle unseen categories.
- Compare three models with five-fold stratified cross-validation on **training data only**
- All models use `class_weight='balanced'` because churners are the minority class; choose by CV recall with F1 tie-break.
- Use `GridSearchCV` on the winning model and evaluate the selected pipeline once
on the held-out test set

| Model | Mean train-fold recall | Mean precision | Mean F1 |
|---|---:|---:|---:|
| **Logistic Regression** | **80.13%** | 51.69% | 62.83% | 
| Decision Tree | 79.06% | 50.03% | 61.25 |
| Random Forest | 64.08% | 59.64% | 61.76% |

**Winning model.** balanced Logistic Regression, tuned `C=0.5`.

**Held-out test:** **78.34% recall**, **50.34% precision**, **61.30% F1**, **73.74% accuracy**.

Confusin matrix (rows = actual Stay/Churn; columns = predicted Stay/Churn):

| | Predicted Stay | Predicted Churn |
|---|---:|---:|
| Actual Stay | 746 | 289 |
| Actual Churn | 81 | 283 |

Interpretation: the model detects 293/374 real churners but flags 289 customers
who actually stayed, so retention incentives should be evaluated against their
cost.

## Business recommendations 
1. Test optional longer-contract incentives with month-to-month customers, measuring incremental retention.
2. Launch a first-year onboarding and proactive support experiment for short-tenure
customers.
3. Offer convenient automated billing to customers paying by electronic check, tracking opt-in and retention rather than assuming causality.
4. Investigate potentially high bills and bills and technical problems through surveys/support trials before designing targeted offers.

## What the app does

- Accepts **all 19 model input features**, including tenure, contract, internet/phone services, payment and charges.
- Scores the customer using the saved `model.joblib` pipeline.
- Displays **Low** (score <35%), **Medium** (35-64%), or **High** (>65%) illustrative risk bands and a 0.50 decision-cutoff classification.
- Suggests customer-service actions based on historical patterns.
- Includes a **Data Explorer** tab with historical churn charts.

**Important:** A balanced classifier's score is not a calibrated probability of future churn. This dataset also does not contain an exact future churn data.

## Files

```text
Customer Churn Predictor/
|-- aap.py
|-- Churn_Project.ipynb
|-- telco_churn.csv
|-- train_model.py
|-- model.joblib
|-- metrics.json
|-- requirements.txt
|-- LEARNING_GUIDE.md
|-- README.md
```

## Run locally

Use Python 3.12 or 3.13, and run from the repository root

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python train_model.py
python -m streamlit runn app.py
```

The saved model is already included. Rerunning `train_model.py` is useful to understand and reproduce how it was made. 

**Dataset:** The Telco Customer Churn CSV supplied with the project brief (commonly shared as the IBM/Kaggle Telco Customer Churn dataset). 

**Tools:** Python, pandas, seaborn, matplotlib, scikit-learn, joblib, Streamlit.