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
    df_clean['benefitsusd2020'] = pd.to_numeric(df_clean['benefitsusd2020'], errors='coerce').fillna(99.5)
    df_clean['totalusd2020'] = pd.to_numeric(df_clean['totalusd2020'], errors='coerce').fillna(152.0)
    df_clean['Country'] = df_clean['Country'].fillna('Unspecified')
    df_clean['Agrofor_practice'] = df_clean['Agrofor_practice'].fillna('Unspecified')
    df_clean['EcC_total_max'] = pd.to_numeric(df_clean['EcC_total_max'], errors='coerce').fillna(0)
    df_clean['EcB_income_max'] = pd.to_numeric(df_clean['EcB_income_max'], errors='coerce').fillna(0)
    df_clean['EcB_tree'] = pd.to_numeric(df_clean['EcB_tree'], errors='coerce').fillna(0)
    return df_clean

try:
    df_clean = load_data()

    rf_model_path = os.path.join(BASE_DIR, "rf_agroforestry_model.joblib" if os.path.exists(os.path.join(BASE_DIR, "rf_agroforestry_model.joblib")) else "rf_agroforestry_model.pkl")
    preprocessor_path = os.path.join(BASE_DIR, "preprocessor.joblib" if os.path.exists(os.path.join(BASE_DIR, "preprocessor.joblib")) else "preprocessor.pkl")
    l1_selector_path = os.path.join(BASE_DIR, "l1_feature_selector.pkl" if os.path.exists(os.path.join(BASE_DIR, "l1_feature_selector.pkl")) else "l1_feature_selector.joblib")

    rf_model = joblib.load(rf_model_path)
    preprocessor = joblib.load(preprocessor_path)
    l1_feature_selector = joblib.load(l1_selector_path)

except Exception as e:
    st.error(f"Error loading assets or data. Please ensure model/data files exist in the repository root: {e}")
    st.stop()

# --- Sidebar Navigation ---
with st.sidebar:
    st.title("Navigation")
    page = st.radio("Go to:", ["📋 Data Overview", "📊 Visualizations", "🔮 Model Prediction"])

    st.markdown("---")
    st.subheader("📁 Model Artifacts")
    st.markdown("""
    - **Model File**: `rf_agroforestry_model.joblib`
    - **Preprocessor Pipeline**: `preprocessor.joblib`
    - **Feature Selection**: `l1_feature_selector.pkl`
    - **R-squared Score**: `~0.7787`
    """)

# --- PAGE 1: DATA OVERVIEW ---
if page == "📋 Data Overview":
    st.title("🌳 Agroforestry Benefits Prediction")
    st.subheader("Project Overview & Data Insights")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Total Observations</div><div class="metric-value">{len(df_clean)}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Unique Countries</div><div class="metric-value">{df_clean["Country"].nunique()}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Median Benefits (USD 2020)</div><div class="metric-value">${df_clean["benefitsusd2020"].median():,.2f}</div></div>', unsafe_allow_html=True)

    st.markdown("### Dataset Preview")
    st.dataframe(df_clean.head(50), use_container_width=True)

# --- PAGE 2: VISUALIZATIONS ---
elif page == "📊 Visualizations":
    st.title("📈 Interactive Data Visualizations")
    tab1, tab2 = st.tabs(["Country Distributions", "Feature Relationships"])

    with tab1:
        st.subheader("Count of Entries per Country")
        fig_country = px.histogram(df_clean, y="Country", color="Country", title="Dataset Volume by Country", height=600)
        fig_country.update_layout(showlegend=False, title_x=0.5)
        st.plotly_chart(fig_country, use_container_width=True)

    with tab2:
        st.subheader("Benefit Distribution (Target Variable)")
        fig_dist = px.histogram(df_clean, x="benefitsusd2020", nbins=30, marginal="rug", title="Distribution of Agroforestry Benefits (USD 2020)", color_discrete_sequence=['#2e6f40'])
        fig_dist.update_layout(title_x=0.5)
        st.plotly_chart(fig_dist, use_container_width=True)

# --- PAGE 3: MODEL PREDICTION ---
elif page == "🔮 Model Prediction":
    st.title("🔮 Forecast Economic Benefits")
    st.write("Enter predictive variables below to estimate predicted agroforestry benefits in USD (2020).")

    with st.form("prediction_form"):
        st.write("#### Input Features")
        col_left, col_right = st.columns(2)

        with col_left:
            totalusd = st.number_input("Total USD 2020 (totalusd2020)", min_value=0.0, value=152.0)
            ecc_total_max = st.number_input("Max Economic Cost (EcC_total_max)", min_value=0.0, value=0.0)
            ecb_income_max = st.number_input("Max Economic Benefit Income (EcB_income_max)", min_value=0.0, value=0.0)
        with col_right:
            ecb_tree = st.number_input("Tree Benefits Value (EcB_tree)", min_value=0.0, value=0.0)
            curr_transplanting = st.selectbox("Transplanting Currency Context", ["N/A", "GH₵ ha-1", "FCFA ha-1", "US$ ha-1"])
            curr_ecc_total = st.selectbox("Max Cost Currency Context", ["N/A", "GH₵ ha-1", "FCFA ha-1", "US$ ha-1"])

        submitted = st.form_submit_button("Predict")

        if submitted:
            try:
                df_template = df_clean.drop(columns=['benefitsusd2020'], errors='ignore').iloc[[0]].copy()

                df_template['totalusd2020'] = float(totalusd)
                df_template['EcC_total_max'] = float(ecc_total_max)
                df_template['EcB_income_max'] = float(ecb_income_max)
                df_template['EcB_tree'] = float(ecb_tree)

                if 'curr_EcC_Transplanting' in df_template.columns:
                    df_template['curr_EcC_Transplanting'] = str(curr_transplanting)
                if 'curr_EcC_total_max' in df_template.columns:
                    df_template['curr_EcC_total_max'] = str(curr_ecc_total)

                input_processed = preprocessor.transform(df_template)
                
                if l1_feature_selector is not None:
                    input_processed = l1_feature_selector.transform(input_processed)

                prediction = rf_model.predict(input_processed)[0]

                st.success(f"### 🎉 Predicted Economic Benefits: ${prediction:,.2f} USD (2020)")
            except Exception as ex:
                st.error(f"Error executing model pipeline preprocessing: {ex}")

# --- Footer ---
st.markdown('<div class="footer">Agroforestry Benefits Analytics Dashboard • Built with Streamlit, Pandas & Scikit-Learn</div>', unsafe_allow_html=True)
