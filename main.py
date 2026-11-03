import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import io
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
    /* Main Background & Fonts */
    .main {
        background-color: #f7f9fb;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1e3d2f;
        color: white;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Header Cards */
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

    /* Title formatting */
    h1, h2, h3 {
        color: #1e3d2f !important;
        font-weight: 700;
    }

    /* Custom Footer */
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

# --- Load Dataset & Trained Models ---
@st.cache_data
def load_data():
    df = pd.read_csv("Costs_and_benefits_of_agroforestry_in_Africa.csv", encoding='latin1')
    df_clean = df.copy()
    df_clean.drop('nbr', axis=1, errors='ignore', inplace=True)
    df_clean['benefitsusd2020'] = pd.to_numeric(df_clean['benefitsusd2020'], errors='coerce').fillna(99.5)
    df_clean['totalusd2020'] = pd.to_numeric(df_clean['totalusd2020'], errors='coerce').fillna(df_clean['totalusd2020'].median() if 'totalusd2020' in df_clean else 152.0)
    df_clean['Country'] = df_clean['Country'].fillna('Unspecified')
    df_clean['Agrofor_practice'] = df_clean['Agrofor_practice'].fillna('Unspecified')
    df_clean['EcC_total_max'] = pd.to_numeric(df_clean['EcC_total_max'], errors='coerce').fillna(0)
    df_clean['EcB_income_max'] = pd.to_numeric(df_clean['EcB_income_max'], errors='coerce').fillna(0)
    df_clean['EcB_tree'] = pd.to_numeric(df_clean['EcB_tree'], errors='coerce').fillna(0)
    return df_clean

try:
    df_clean = load_data()
    
    # Dynamic loading of saved joblib and pickle models with robustness
    rf_model_path = "rf_agroforestry_model.joblib" if os.path.exists("rf_agroforestry_model.joblib") else "rf_agroforestry_model.pkl"
    preprocessor_path = "preprocessor.joblib" if os.path.exists("preprocessor.joblib") else "preprocessor.pkl"
    l1_selector_path = "l1_feature_selector.pkl" if os.path.exists("l1_feature_selector.pkl") else "l1_feature_selector.joblib"
    
    rf_model = joblib.load(rf_model_path)
    preprocessor = joblib.load(preprocessor_path)
    l1_feature_selector = joblib.load(l1_selector_path)

except Exception as e:
    st.error(f"Error loading assets or data. Please ensure model/data files exist: {e}")
    st.stop()

# --- Sidebar Navigation ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/616/616412.png", width=100)
    st.title("Navigation")
    page = st.radio("Go to:", ["📋 Data Overview", "📊 Visualizations", "🔮 Model Prediction"])

    st.markdown("---")
    st.subheader("📂 Model Artifacts & Resources")
    st.markdown("""
    - **Model File**: `rf_agroforestry_model.joblib`
    - **Preprocessor Pipeline**: `preprocessor.joblib`
    - **Feature Selection Model**: `l1_feature_selector.pkl`
    - **R-squared Score**: `~0.7787` (Tuned Test Set)
    - **Primary Features**: `totalusd2020`, `EcC_total_max`, `EcB_income_max`, `EcB_tree`
    - **Data Preprocessing**: Standard Scaler, One-Hot Encoding, L1 Regularization Selectors
    """)

# --- PAGE 1: DATA OVERVIEW ---
if page == "📋 Data Overview":
    st.title("🌳 Agroforestry Benefits Prediction")
    st.subheader("Project Overview & Data Insights")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class=
