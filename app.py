import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="Agroforestry Analytics Dashboard",
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
    # Load saved model objects with fallback options
    try:
        rf_model = joblib.load("rf_agroforestry_model.joblib")
    except:
        rf_model = joblib.load("rf_agroforestry_model.pkl")

    try:
        preprocessor = joblib.load("preprocessor.joblib")
    except:
        preprocessor = joblib.load("preprocessor.pkl")

    try:
        l1_feature_selector = joblib.load("l1_feature_selector.joblib")
    except:
        l1_feature_selector = joblib.load("l1_feature_selector.pkl")
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
    - **Feature Selection Model**: `l1_feature_selector.joblib`
    - **Feature Metadata**: `top_features.txt`
    - **R-squared Score**: `~0.7787` (Tuned Test Set)
    - **Primary Features**: `totalusd2020`, `EcC_total_max`, `EcB_income_max`, `EcB_tree`
    - **Data Preprocessing**: Standard Scaler, One-Hot Encoding, L1 Regularization Selectors
    """)

# --- PAGE 1: DATA OVERVIEW ---
if page == "📋 Data Overview":
    st.title("🌳 Agroforestry Benefits Dashboard")
    st.subheader("Project Overview & Data Insights")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-title">Total Observations</div><div class="metric-value">{}</div></div>'.format(len(df_clean)), unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-title">Unique Countries</div><div class="metric-value">{}</div></div>'.format(df_clean['Country'].nunique()), unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-title">Median Benefits (USD 2020)</div><div class="metric-value">${:,.2f}</div></div>'.format(df_clean['benefitsusd2020'].median()), unsafe_allow_html=True)

    st.markdown("### Dataset Preview")
    st.dataframe(df_clean.head(50), use_container_width=True)

    st.markdown("### Key Feature Description (RFE Top Selected Features)")
    st.write("Our models utilize the top Recursive Feature Elimination (RFE) features containing variables such as Total Cost/Benefit values (totalusd2020, EcC_total_max, EcB_income_max, EcB_tree) optimized with Lasso Regularization.")

# --- PAGE 2: VISUALIZATIONS ---
elif page == "📊 Visualizations":
    st.title("📈 Interactive Data Visualizations")
    st.write("Explore dataset correlations, distribution densities, and macro trends.")

    tab1, tab2 = st.tabs(["Country Distributions", "Feature Relationships"])

    with tab1:
        st.subheader("Total Benefits by Agroforestry Practice per Country")
        df_grouped = df_clean.groupby(['Country', 'Agrofor_practice'])['benefitsusd2020'].sum().reset_index()
        stacked_bar_fig = px.bar(
            df_grouped,
            x='Country',
            y='benefitsusd2020',
            color='Agrofor_practice',
            title='Total Benefits (USD 2020) per Country',
            barmode='stack'
        )
        st.plotly_chart(stacked_bar_fig, use_container_width=True)

        st.subheader("Dataset Volume by Country")
        country_counts = df_clean['Country'].value_counts().reset_index()
        country_counts.columns = ['Country', 'Count']
        fig_country = px.bar(country_counts, x='Count', y='Country', orientation='h', color='Count', color_continuous_scale='Viridis')
        st.plotly_chart(fig_country, use_container_width=True)

    with tab2:
        st.subheader("Benefit Distribution (Target Variable)")
        fig_dist = px.histogram(df_clean, x='benefitsusd2020', nbins=30, marginal='rug', color_discrete_sequence=['#2e6f40'])
        st.plotly_chart(fig_dist, use_container_width=True)

        st.subheader("Relationship: totalusd2020 vs Benefits")
        fig_scatter = px.scatter(df_clean, x='totalusd2020', y='benefitsusd2020', hover_name='Country', color='benefitsusd2020', color_continuous_scale='Aggrnyl')
        st.plotly_chart(fig_scatter, use_container_width=True)

# --- PAGE 3: MODEL PREDICTION ---
elif page == "🔮 Model Prediction":
    st.title("🔮 Forecast Benefits")
    st.write("Specify model variables dynamically below to estimate predicted benefits in USD (2020).")

    with st.form("prediction_form"):
        st.write("#### Enter Predictor Values")
        c1, c2 = st.columns(2)

        with c1:
            totalusd = st.number_input("Total USD 2020 (totalusd2020)", min_value=0.0, value=152.0)
            ecc_total_max = st.number_input("Max Economic Cost (EcC_total_max)", min_value=0.0, value=0.0)
            ecb_income_max = st.number_input("Max Economic Benefit Income (EcB_income_max)", min_value=0.0, value=0.0)
        with c2:
            ecb_tree = st.number_input("Tree Benefits Value (EcB_tree)", min_value=0.0, value=0.0)
            curr_transplanting = st.selectbox("Transplanting Currency Context", ["N/A", "GH? ha-1", "FCFA ha-1", "US$ ha-1"])
            curr_ecc_total = st.selectbox("Max Cost Currency Context", ["N/A", "GH? ha-1", "FCFA ha-1", "US$ ha-1"])

        submitted = st.form_submit_button("Run Model Prediction")

        if submitted:
            # Assemble dynamic row templates matching preprocessor requirements
            df_template = df_clean.drop(columns=['benefitsusd2020'], errors='ignore').iloc[[0]].copy()

            # Overwrite values manually mapped
            df_template['totalusd2020'] = totalusd
            df_template['EcC_total_max'] = ecc_total_max
            df_template['EcB_income_max'] = ecb_income_max
            df_template['EcB_tree'] = ecb_tree
            if 'curr_EcC_Transplanting' in df_template.columns:
                df_template['curr_EcC_Transplanting'] = curr_transplanting
            if 'curr_EcC_total_max' in df_template.columns:
                df_template['curr_EcC_total_max'] = curr_ecc_total

            try:
                input_processed = preprocessor.transform(df_template)
                input_processed_L1 = l1_feature_selector.transform(input_processed)
                prediction = rf_model.predict(input_processed_L1)[0]

                st.success(f"### 🎉 Predicted Economic Benefits: ${prediction:,.2f} USD (2020)")
            except Exception as ex:
                st.error(f"Error executing model pipeline preprocessing: {ex}")

# --- Global Footer ---
st.markdown('<div class="footer">Agroforestry Benefits Analytics Dashboard • Built with Streamlit, Pandas & Scikit-Learn</div>', unsafe_allow_html=True)
