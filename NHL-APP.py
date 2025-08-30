import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score,
                             classification_report, confusion_matrix)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
import os

# --------------------
# Page Config
# --------------------
st.set_page_config(page_title="🏥 NHI in SA — Dashboard", page_icon="🏥", layout="wide")

# --------------------
# Helper Functions
# --------------------
@st.cache_data
def read_csv(file):
    return pd.read_csv(file)

def kpi_card(col, label, value):
    col.metric(label, value)

def safe_parse_dates(df, date_cols=("date","appointment_date","visit_date","created_at")):
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def build_sample_data(n_patients=500, seed=42):
    rng = np.random.default_rng(seed)
    provinces = ["Gauteng","KwaZulu-Natal","Western Cape","Eastern Cape","Limpopo","Mpumalanga","North West","Free State","Northern Cape"]
    sexes = ["Male","Female"]
    specialties = ["GP","Pediatrics","Cardiology","Orthopedics","OBGYN"]
    facility_types = ["Clinic","District Hospital","Regional Hospital","Private Hospital"]

    patients = pd.DataFrame({
        "patient_id": np.arange(1, n_patients+1),
        "age": rng.integers(0, 90, size=n_patients),
        "sex": rng.choice(sexes, size=n_patients),
        "province": rng.choice(provinces, size=n_patients),
        "income_bracket": rng.choice(["Low","Middle","High"], size=n_patients, p=[0.55, 0.35, 0.10]),
    })

    n_docs = 120
    doctors = pd.DataFrame({
        "doctor_id": np.arange(1, n_docs+1),
        "specialty": rng.choice(specialties, size=n_docs),
        "facility_type": rng.choice(facility_types, size=n_docs),
        "province": rng.choice(provinces, size=n_docs),
    })

    n_appts = n_patients * 3
    appt_dates = pd.date_range("2023-01-01", "2025-08-01", freq="D")
    appointments = pd.DataFrame({
        "appointment_id": np.arange(1, n_appts+1),
        "patient_id": rng.integers(1, n_patients+1, size=n_appts),
        "doctor_id": rng.integers(1, n_docs+1, size=n_appts),
        "appointment_date": rng.choice(appt_dates, size=n_appts),
        "status": rng.choice(["attended", "no_show"], size=n_appts, p=[0.85, 0.15]),
    })

    base_cost = {"Clinic": (150,400), "District Hospital": (300,1200),
                 "Regional Hospital": (600,2500), "Private Hospital": (1200,7000)}

    merged = appointments.merge(doctors[["doctor_id","facility_type"]], on="doctor_id", how="left")
    lo, hi = zip(*[base_cost.get(ft, (200,2000)) for ft in merged["facility_type"].fillna("Clinic")])
    amounts = np.random.default_rng(seed+1).uniform(np.array(lo), np.array(hi))
    billing = merged[["appointment_id"]].copy()
    billing["amount"] = amounts.round(2)

    return patients, appointments, billing, doctors

def merge_tables(patients, appointments, billing, doctors):
    df = appointments.merge(patients, on="patient_id", how="left") \
                     .merge(doctors, on="doctor_id", how="left") \
                     .merge(billing, on="appointment_id", how="left")
    df = safe_parse_dates(df)
    return df

# --------------------
# Sidebar — Data Input
# --------------------
st.sidebar.title("📥 Data Input")
st.sidebar.caption("Upload your 4 CSV tables or use synthetic sample")

use_sample = st.sidebar.checkbox("Use Sample Data", value=True)
model_file = st.sidebar.text_input("Model File Name", value="model.pkl")

if use_sample:
    patients_df, appointments_df, billing_df, doctors_df = build_sample_data()
else:
    up_patients = st.sidebar.file_uploader("Upload patients.csv", type="csv")
    up_appointments = st.sidebar.file_uploader("Upload appointments.csv", type="csv")
    up_billing = st.sidebar.file_uploader("Upload billing.csv", type="csv")
    up_doctors = st.sidebar.file_uploader("Upload doctors.csv", type="csv")

    if not all([up_patients, up_appointments, up_billing, up_doctors]):
        st.warning("Upload all 4 CSV files or tick 'Use Sample Data'.")
        st.stop()

    patients_df = read_csv(up_patients)
    appointments_df = read_csv(up_appointments)
    billing_df = read_csv(up_billing)
    doctors_df = read_csv(up_doctors)

df = merge_tables(patients_df, appointments_df, billing_df, doctors_df)

# --------------------
# Tabs
# --------------------
explore_tab, model_tab = st.tabs(["📊 Explore", "🤖 Model"])

# --------------------
# Explore Tab
# --------------------
with explore_tab:
    st.subheader("Data Overview")
    c1, c2, c3 = st.columns(3)
    kpi_card(c1, "Patients", df['patient_id'].nunique())
    kpi_card(c2, "Visits", df['appointment_id'].nunique())
    kpi_card(c3, "Total Spend (R)", round(df['amount'].sum(),2))

    st.markdown("#### Visits by Province")
    by_prov = df.groupby("province")["appointment_id"].nunique().reset_index(name="visits")
    fig1 = px.bar(by_prov, x="province", y="visits")
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("#### Monthly Visit Trend")
    df["year_month"] = df["appointment_date"].dt.to_period("M").astype(str)
    trend = df.groupby("year_month")["appointment_id"].nunique().reset_index(name="visits")
    fig2 = px.line(trend, x="year_month", y="visits")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Data Preview")
    st.dataframe(df.head(50))

# --------------------
# Model Tab
# --------------------
with model_tab:
    st.subheader("ML Model & Prediction")

    # Load model if exists
    if os.path.exists(linear_regression_model.pkl):
        pipe = joblib.load(linear_regression_model.pkl)
        st.success(f"Loaded model from {linear_regression_model.pkl}")
    else:
        pipe = None
        st.info("Train a model first.")

    st.markdown("#### Quick Prediction")
    if pipe:
        input_data = {}
        for col in pipe.named_steps['prep'].transformers_[0][2]:  # numeric
            input_data[col] = st.number_input(col, value=0.0)
        for col in pipe.named_steps['prep'].transformers_[1][2]:  # categorical
            input_data[col] = st.selectbox(col, ["Unknown"])

        if st.button("Predict"):
            row = pd.DataFrame([input_data])
            pred = pipe.predict(row)[0]
            st.success(f"Prediction: {pred}")
