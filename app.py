import os
import traceback
import warnings

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import sklearn

from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

warnings.filterwarnings("ignore")

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Agroforestry Benefits Prediction",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #f7f9fb;
        font-family: Arial, sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: #1e3d2f;
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
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 3. FILE PATHS
# Put app.py, the CSV, and the three model files in one folder.
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def find_file(*filenames):
    for filename in filenames:
        path = os.path.join(BASE_DIR, filename)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
