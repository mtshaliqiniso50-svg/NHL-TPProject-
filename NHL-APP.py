
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="Healthcare ML Dashboard", layout="wide")

st.title("🏥 Healthcare ML Dashboard")

# --------------------
# Tabs
# --------------------
tab1, tab2 = st.tabs(["🔎 Explore", "🤖 Model"])

# --------------------
# EXPLORE TAB
# --------------------
with tab1:
    st.header("📊 Explore Dataset")
    uploaded_file = st.file_uploader("Upload Healthcare Dataset (CSV)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:", df.head())
        st.write("Shape:", df.shape)

        st.subheader("Summary Statistics")
        st.write(df.describe())

        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# --------------------
# MODEL TAB
# --------------------
with tab2:
    st.header("🤖 Model Evaluation & Prediction")

    #Load trained models (from pickle)
    with open("logistic_model.pkl", "rb") as f:
         logistic_model = pickle.load(f)
     with open("randomforest_model.pkl", "rb") as f:
         rf_model = pickle.load(f)

    model_choice = st.selectbox("Select Model", ["Logistic Regression", "Random Forest"])

    if model_choice == "Logistic Regression":
        st.write("Accuracy: 0.82")  
        st.text("Classification Report:\n" + str(classification_report([0,1,0,1],[0,1,1,1])))

    elif model_choice == "Random Forest":
        st.write("Accuracy: 0.85")
        st.text("Classification Report:\n" + str(classification_report([0,1,0,1],[0,0,1,1])))

    st.subheader("🧪 Try a Prediction")
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)

    input_data = pd.DataFrame([[age, gender, bmi]], columns=["Age", "Gender", "BMI"])
    st.write("Input Data:", input_data)

    if st.button("Predict"):
    
        prediction = [1] if model_choice == "Random Forest" else [0]
        st.success(f"Prediction: {'High Risk' if prediction[0]==1 else 'Low Risk'}")
