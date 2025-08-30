import streamlit as st
import pandas as pd
import streamlit as st
import pickle
import pandas as pd

# Load models once at startup
model_files = [
    "baseline_model.pkl",
    "linear_regression_model.pkl",
    "random_forest_model.pkl",
    "gradient_boosting_model.pkl"
]
models = {}
for filename in model_files:
    with open(filename, "rb") as f:
        models[filename] = pickle.load(f)
st.set_page_config(page_title="Healthcare ML Dashboard", layout="wide")
st.title("🏥 Healthcare ML Dashboard")
