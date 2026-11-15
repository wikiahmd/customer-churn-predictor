from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Telco Retention Lab", page_icon="📞", layout="wide")

@st.cache_resource
def load_model():
    return  joblib.load(ROOT / "model.joblib")

@st.cache_data
def load_data():
    df = pd.read_csv(ROOT / "telco_churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df.tenure.eq(0) & df.TotalCharges.isna(), "TotalCharges"] = 0.0
    return df

model = load_model()
metrics = json.loads((ROOT / "metrics.json").read_text(encoding="utf-8"))
st.title("📞 Telco Retention Lab")
st.caption("Explore historical churn pattern, score a customer and find possible retention actions.")

predict_tab, insights_tab = st.tabs(["🔎 Predict churn risk", "📊 Explore the data"])

with predict_tab:
    st.write("Enter one customer's details. The classifier uses all 19 training features, not just billing information.")
    with st.form("churn_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Customer")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", ["No", "Yes"])
            partner = st.selectbox("Has partner", ["No", "Yes"])
            dependents = st.selectbox("Has dependents", ["No", "Yes"])
            tenure = st.number_input("Tenure (months)", 0, 72, 10, 1)
            st.subheader("Contract")
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
            payment = st.selectbox("Payment method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
        with col2:
            st.subheader("Services")
            phone = st.selectbox("Phone service", ["Yes", "No"])
            multiple = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
            internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
            online_security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox('Device protection', ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech support", ["No", "Yes", "No interent service"])
        with col3:
            st.subheader("Usage and charges")
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])
            monthly = st.number_input("Monthly charges ($)", min_value=0.0, max_value=200.0, value=75.0, step=1.0)
            total = st.number_input("Total charges to date ($)", min_value=0.0, value=750.0, step=10.0,
                                    help="Enter the real accumulated charges, not the monthly bill. For tenure 0, use 0.")
            st.info("Tip: Try changing the contract and tenure to see how the model's score changes. The score is not a causal effect.")
        submitted = st.form_submit_button("Calculate churn risk", type="primary", use_container_width=True)

    if submitted:
        if phone == "No":
            multiple = "No phone service"
        elif multiple == "No phone service":
            multiple = "No"
        internet_dependent = [online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies]
        if internet == "No":
            internet_dependent = ["No internet service"] * 6
        else:
            internet_dependent = ["No" if value == "No internet service" else value for value in internet_dependent]

        (
            online_security, 
            online_backup, 
            device_protection,
            tech_support, 
            streaming_tv, 
            streaming_movies
            ) = internet_dependent
        
        if tenure == 0:
            total = 0.0

        customer = {
            "gender": gender, "SeniorCitizen": int(senior == "Yes"), "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
            "MultipleLines": multiple, "InternetService": internet,
            "OnlineSecurity": online_security, "OnlineBackup": online_backup,
            "DeviceProtection": device_protection, "TechSupport": tech_support,
            "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
            "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly, "TotalCharges": total,
        }

        if not hasattr(model, "feature_names_in"):
            st.error(
                "The saved model has no feature_names_in_. Check that you saved "
                "the full fitted preprocessing + classification pipeline and "
                "supply the exact feature order used at training."
            )
            st.stop()

        feature_names = list(model.feature_names_in_)
        missing = set(feature_names) - set(customer)
        if missing:
            st.error(f"The trained model expects features not provided by this app: {sorted(missing)}")
            st.stop()

        record = pd.DataFrame([customer])[feature_names]
        classes = list(model.classes_)
        churn_class = next((c for c in classes if c == 1 or c== "Yes"), None)
        if churn_class is None:
            st.error(f"Cannot identify the churn class in the model's classes_: {classes}")
            st.stop()
        score = float(model.predict_proba(record)[0, classes.index(churn_class)])
        predicted_churn = score >= 0.50
        risk = "High" if score >= 0.65 else "Medium" if score >= 0.35 else "Low"

        st.divider()
        a, b, c = st.columns(3)
        a.metric("Risk category", risk)
        b.metric("Model score", f"{score:.1%}")
        c.metric("Prediction (0.50 cutoff)", "Churn" if predicted_churn else "Stay")
        st.progress(score)
        st.caption("Risk category bands: Low <35%, Medium 35-64%, High >65%. These are illustrative model-score bands, not calibrated probabilities. Predictions are educational, not guarantess.")

        st.subheader("Possible retention actions")
        actions = []
        if contract == "Month-to-month":
            actions.append("Test an optional discount for switching from month-to-month to a longer contract.")
        if tenure <= 12:
            actions.append("Offer first-year onboarding and a proactive service check-in.")
        if payment == "Electronic check":
            actions.append("Offer a friction-free automatic payment option; test whether it improves retention.")
        if internet != "No" and tech_support == "No":
            actions.append("Offer a technical support trial and address unresolved service issues.")
        if monthly >= 85:
            actions.append("Review the bill with the customer and discuss a lower-cost plan if suitable.")
        if not actions:
            actions.append("Maintain service quality and survey the customer before proposing a tailored incentive.")
        for action in actions:
            st.markdown(f"- {action}")
        st.caption("These actions are hypotheses based on group associations, not proven causes of churn. Test offers through experiments.")

with insights_tab:
    df = load_data()
    df["ChurnFlag"] = df["Churn"].eq("Yes").astype(int)
    st.subheader("What does the historical dataset show?")
    n1, n2 = st.columns(2)
    n1.metric("Customers", f"{len(df):,}")
    n2.metric("Observed churn", f"{df['ChurnFlag'].mean():.1%}")
    left, right = st.columns(2)
    with left:
        st.write("**Churn rate by contract**")
        st.bar_chart((df.groupby("Contract")["ChurnFlag"].mean() * 100).rename("Churn %"), y_label="Churn (%)")
        st.write("**Churn rate by payment method**")
        st.bar_chart((df.groupby("PaymentMethod")["ChurnFlag"].mean() * 100).rename("Churn %"), y_label="Churn (%)")
    with right:
        grouped = df.assign(TenureGroup=pd.cut(df.tenure, bins=[-1, 12, 24, 48, 72], labels=["0-12", "13-24", "25-48", "49-72"]))
        st.write("**Churn by tenure (months)**")
        tenure_pct = (grouped.groupby("TenureGroup", observed=True)["ChurnFlag"].mean() * 100).rename("Churn %")
        tenure_pct.index = tenure_pct.index.astype(str)
        st.bar_chart(tenure_pct, y_label="Churn (%)")
        grouped = df.assign(BillGroup=pd.qcut(df.MonthlyCharges, q=4, duplicates="drop"))
        st.write("**Churn by monthly bill quartile**")
        bill_pct = (grouped.groupby("BillGroup", observed=True)["ChurnFlag"].mean() * 100).rename("Churn %")
        bill_pct.index = bill_pct.index.astype(str)
        st.bar_chart(bill_pct, y_label="Churn (%)")
    st.divider()
    st.write(f"**Trained model:** {metrics['selected_model']} | **held-out recall:** {metrics['test_metrics']['recall']:.1%} | **held-out precision:** {metrics['test_metrics']['precision']:.1%}")
    st.caption("This cross-sectional dataset supports associations and model validation, not causal explanations or am exact date when a customer will leave.")

