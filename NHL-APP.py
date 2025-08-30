import streamlit as st
import pandas as pd
import streamlit as st
import pickle

# Load models once at startup
model_file = [
    "model.pkl"
]
models = {}
for filename in model_file:
    with open(filename, "rb") as f:
        models[filename] = pickle.load(f)
st.set_page_config(page_title="Healthcare ML Dashboard", layout="wide")
st.title("🏥 Healthcare ML Dashboard")
