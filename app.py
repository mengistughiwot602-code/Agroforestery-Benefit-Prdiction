import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import io
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Agroforestry Analytics",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    /* Main background */
    .main {
        background-color: #f4f7f6;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Sidebar customization */
    [data-testid="stSidebar"] {
        background-image: linear-gradient(180deg, #112d1b 0%, #1e3d2f 100%);
        color: white;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Styled Cards for KPI metrics */
    .kpi-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        border-left: 6px solid #2a7e43;
        margin-bottom: 15px;
    }
    .kpi-title {
        font-size: 14px;
        color: #7f8c8d;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    .kpi-value {
        font-size: 26px;
        color: #2c3e50;
        font-weight: bold;
        margin-top: 5px;
    }
    
    /* App titles */
    .main-title {
        color: #112d1b;
        font-weight: 800;
        font-size: 32px;
        margin-bottom: 5px;
    }
    .sub-title {
        color: #2a7e43;
        font-weight: 600;
        font-size: 18px;
        margin-bottom: 25px;
    }
</style>
""", unsafe_allow_html=True)

# --- Loading Data and Models Safely ---
@st.cache_data
def load_data():
    df = pd.read_csv("Costs_and_benefits_of_agroforestry_in_Africa.csv", encoding='latin1')
    df_clean = df.copy()
    df_clean.drop('nbr', axis=1, errors='ignore', inplace=True)
    # Ensure target/key financial values are numeric without raising errors
    for col in ['benefitsusd2020', 'totalusd2020', 'EcC_total_max', 'EcB_income_max', 'EcB_tree']:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
    return df_clean

try:
    df_clean = load_data()
    
    # Dynamic loading of saved joblib and pickle models
    rf_model = joblib.load("rf_agroforestry_model.joblib")
    preprocessor = joblib.load("preprocessor.joblib")
    l1_feature_selector = joblib.load("l1_feature_selector.joblib")
    
    # Get original features (X) template used in training
    target_col = "benefitsusd2020"
    X_template = df_clean.drop(columns=[target_col, "spacing_btw_trees", "benefitsmy"], errors="ignore")
except Exception as e:
    st.error(f"Initialization error: {e}. Please run the Colab notebook cells first to generate required artifacts.")
    st.stop()

# --- Sidebar Navigation ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>🌴 MENU 🌴</h2>", unsafe_allow_html=True)
    page = st.radio("Jump to section:", ["📋 Overview", "📊 Analytics", "🔮 Predict Benefits"])
    
    st.markdown("---")
    st.subheader("⚡ Artifacts Loaded")
    st.info("✅ Random Forest Regressor\n✅ Preprocessing Pipeline\n✅ L1 Lasso Selector")

# --- PAGE 1: OVERVIEW ---
if page == "📋 Overview":
    st.markdown("<div class='main-title'>🌳 Agroforestry Benefits Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Overview of the Agroforestry Cost & Benefit Study in Africa</div>", unsafe_allow_html=True)
    
    # Interactive Cards Row
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Dataset Records</div><div class="kpi-value">{len(df_clean)}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Participating Countries</div><div class="kpi-value">{df_clean["Country"].nunique()}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Avg Benefits (USD 2020)</div><div class="kpi-value">${df_clean["benefitsusd2020"].mean():,.2f}</div></div>', unsafe_allow_html=True)
        
    st.markdown("### Sample Data Snapshot")
    st.dataframe(df_clean.head(20), use_container_width=True)

# --- PAGE 2: ANALYTICS ---
elif page == "📊 Analytics":
    st.markdown("<div class='main-title'>📊 Interactive Visualizations</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Explore Agroforestry Trends & Metrics</div>", unsafe_allow_html=True)
    
    col_plot1, col_plot2 = st.columns(2)
    
    with col_plot1:
        fig1 = px.histogram(df_clean, x="Country", title="Records Count per Country", color="Country", color_discrete_sequence=px.colors.sequential.Teal)
        st.plotly_chart(fig1, use_container_width=True)
        
    with col_plot2:
        fig2 = px.scatter(df_clean, x="totalusd2020", y="benefitsusd2020", color="Country", title="Total Cost vs. Benefits (USD 2020)", hover_data=["Agrofor_practice"])
        st.plotly_chart(fig2, use_container_width=True)

# --- PAGE 3: MODEL PREDICTION ---
elif page == "🔮 Predict Benefits":
    st.markdown("<div class='main-title'>🔮 Predict Benefits</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Enter biophysical and economic values to calculate predicted benefits</div>", unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        st.write("##### Fill details:")
        c_l, c_r = st.columns(2)
        
        with c_l:
            totalusd2020 = st.number_input("Total USD 2020 (totalusd2020)", min_value=0.0, value=152.0)
            EcC_total_max = st.number_input("Max Economic Cost (EcC_total_max)", min_value=0.0, value=0.0)
            EcB_income_max = st.number_input("Max Economic Benefit Income (EcB_income_max)", min_value=0.0, value=0.0)
        with c_r:
            EcB_tree = st.number_input("Tree Benefits Value (EcB_tree)", min_value=0.0, value=0.0)
            curr_transplanting = st.selectbox("Transplanting Currency Context", ["N/A", "GH? ha-1", "FCFA ha-1", "US$ ha-1"])
            curr_ecc_total = st.selectbox("Max Cost Currency Context", ["N/A", "GH? ha-1", "FCFA ha-1", "US$ ha-1"])
            
        submit_btn = st.form_submit_button("Run Estimation Model")
        
        if submit_btn:
            try:
                # Generate template row matching the training set columns
                input_row = X_template.iloc[[0]].copy()
                
                # Carefully map user input ensuring numeric columns remain numeric
                input_row['totalusd2020'] = float(totalusd2020)
                input_row['EcC_total_max'] = float(EcC_total_max)
                input_row['EcB_income_max'] = float(EcB_income_max)
                input_row['EcB_tree'] = float(EcB_tree)
                
                if 'curr_EcC_Transplanting' in input_row.columns:
                    input_row['curr_EcC_Transplanting'] = str(curr_transplanting)
                if 'curr_EcC_total_max' in input_row.columns:
                    input_row['curr_EcC_total_max'] = str(curr_ecc_total)
                
                # Apply transformations sequentially
                input_processed = preprocessor.transform(input_row)
                input_processed_l1 = l1_feature_selector.transform(input_processed)
                prediction = rf_model.predict(input_processed_l1)[0]
                
                st.balloons()
                st.success(f"### 🎉 Predicted Agroforestry Economic Benefits: ${prediction:,.2f} USD (2020)")
            except Exception as ex:
                st.error(f"Error processing model predictions: {ex}")
