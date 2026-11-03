import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Agroforestry Benefits Prediction",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS Styling ---
st.markdown("""
<style>
    .main {
        background-color: #f7f9fb;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: #1e3d2f;
        color: white;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    .metric-card {
        background-color: white;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border-left: 5px solid #2e6f40;
        margin-bottom: 20px;
    }
    .metric-title {
        font-size: 14px;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: bold;
    }
    .metric-value {
        font-size: 28px;
        color: #1e3d2f;
        font-weight: 700;
        margin-top: 8px;
    }
    h1, h2, h3 {
        color: #1e3d2f !important;
        font-weight: 700;
    }
    .footer {
        text-align: center;
        padding: 20px;
        font-size: 12px;
        color: #888888;
        border-top: 1px solid #e0e0e0;
        margin-top: 50px;
    }
</style>
""", unsafe_allow_html=True)

# --- Base Directory Pathing ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Load Dataset & Trained Models with Robust Paths ---
@st.cache_data
def load_data():
    csv_path = os.path.join(BASE_DIR, "Costs_and_benefits_of_agroforestry_in_Africa.csv")
    df = pd.read_csv(csv_path, encoding='latin1')
    df_clean = df.copy()
    df_clean = df_clean.drop(columns=['nbr'], errors='ignore')
    
    # Clean numeric columns cleanly
    for col in df_clean.columns:
        if col not in ['Country', 'Agrofor_practice', 'curr_EcC_Transplanting', 'curr_EcC_total_max', 'curr_EcB_income_max', 'curr_EcB_tree']:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    df_clean['benefitsusd2020'] = df_clean['benefitsusd2020'].fillna(99.5)
    df_clean['totalusd2020'] = df_clean['totalusd2020'].fillna(152.0)
    df_clean['Country'] = df_clean['Country'].fillna('Unspecified').astype(str)
    df_clean['Agrofor_practice'] = df_clean['Agrofor_practice'].fillna('Unspecified').astype(str)
    df_clean['EcC_total_max'] = df_clean['EcC_total_max'].fillna(0.0)
    df_clean['EcB_income_max'] = df_clean['EcB_income_max'].fillna(0.0)
    df_clean['EcB_tree'] = df_clean['EcB_tree'].fillna(0.0)
    return df_clean

try:
    df_clean = load_data()

    rf_model_path = os.path.join(BASE_DIR, "rf_agroforestry_model.joblib" if os.path.exists(os.path.join(BASE_DIR, "rf_agroforestry_model.joblib")) else "rf_agroforestry_model.pkl")
    preprocessor_path = os.path.join(BASE_DIR, "preprocessor.joblib" if os.path.exists(os.path.join(BASE_DIR, "preprocessor.joblib")) else "preprocessor.pkl")
