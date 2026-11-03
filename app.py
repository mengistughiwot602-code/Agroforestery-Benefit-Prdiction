
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Agroforestry Benefits Prediction",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 2. CUSTOM CSS
# ============================================================

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


# ============================================================
# 3. BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# 4. DATASET LOADING AND CLEANING
# ============================================================

@st.cache_data
def load_data():

    csv_path = os.path.join(
        BASE_DIR,
        "Costs_and_benefits_of_agroforestry_in_Africa.csv"
    )

    df = pd.read_csv(csv_path, encoding="latin1")

    df_clean = df.copy()

    df_clean = df_clean.drop(
        columns=["nbr"],
        errors="ignore"
    )

    categorical_columns = [
        "Country",
        "Agrofor_practice",
        "curr_EcC_Transplanting",
        "curr_EcC_total_max",
        "curr_EcB_income_max",
        "curr_EcB_tree"
    ]

    # Convert non-categorical columns into numeric values.
    for col in df_clean.columns:

        if col not in categorical_columns:

            df_clean[col] = pd.to_numeric(
                df_clean[col],
                errors="coerce"
            )

    # Handle missing target and cost values.
    if "benefitsusd2020" in df_clean.columns:
        df_clean["benefitsusd2020"] = (
            df_clean["benefitsusd2020"].fillna(99.5)
        )

    if "totalusd2020" in df_clean.columns:
        df_clean["totalusd2020"] = (
            df_clean["totalusd2020"].fillna(152.0)
        )

    # Handle categorical columns.
    for col in categorical_columns:

        if col in df_clean.columns:

            df_clean[col] = (
                df_clean[col]
                .fillna("Unspecified")
                .astype(str)
            )

    # Handle important numerical variables.
    for col in [
        "EcC_total_max",
        "EcB_income_max",
        "EcB_tree"
    ]:

        if col in df_clean.columns:

            df_clean[col] = (
                df_clean[col]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )

    return df_clean


# ============================================================
# 5. LOAD MODEL ARTIFACTS
# ============================================================

def find_model_file(*filenames):

    for filename in filenames:

        path = os.path.join(BASE_DIR, filename)

        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"None of these model files were found: {filenames}"
    )


try:

    df_clean = load_data()

    rf_model_path = find_model_file(
        "rf_agroforestry_model.joblib",
        "rf_agroforestry_model.pkl"
    )

    preprocessor_path = find_model_file(
        "preprocessor.joblib",
        "preprocessor.pkl"
    )

    l1_selector_path = find_model_file(
        "l1_feature_selector.pkl",
        "l1_feature_selector.joblib"
    )

    rf_model = joblib.load(rf_model_path)

    preprocessor = joblib.load(preprocessor_path)

    l1_feature_selector = joblib.load(l1_selector_path)

except Exception as e:

    st.error(
        "Error loading assets or data. "
        "Please ensure all model and dataset files exist "
        f"in the repository root: {e}"
    )

    st.stop()


# ============================================================
# 6. PREDICTION PREPROCESSING FUNCTION
# ============================================================

def prepare_prediction_input(df_input, df_reference, preprocessor):

    df_input = df_input.copy()

    # Use the exact feature names expected by the fitted preprocessor.
    expected_columns = getattr(
        preprocessor,
        "feature_names_in_",
        None
    )

    if expected_columns is not None:

        expected_columns = list(expected_columns)

        missing_columns = [
            col for col in expected_columns
            if col not in df_input.columns
        ]

        if missing_columns:

            raise ValueError(
                "Missing features required by the trained model: "
                f"{missing_columns}"
            )

        df_input = df_input.reindex(
            columns=expected_columns
        )

    # Identify numerical columns expected by the fitted
    # ColumnTransformer, where possible.
    numeric_model_columns = set()

    transformers = getattr(
        preprocessor,
        "transformers_",
        []
    )

    for transformer_name, transformer, selected_columns in transformers:

        if transformer_name == "remainder":
            continue

        if isinstance(transformer, str):
            continue

        if isinstance(selected_columns, slice):

            selected = list(
                df_input.columns[selected_columns]
            )

        elif isinstance(
            selected_columns,
            (list, tuple, np.ndarray, pd.Index)
        ):

            selected = list(selected_columns)

            if selected and isinstance(
                selected[0],
                (bool, np.bool_)
            ):

                selected = list(
                    df_input.columns[
                        np.asarray(selected, dtype=bool)
                    ]
                )

            elif selected and isinstance(
                selected[0],
                (int, np.integer)
            ):

                selected = [
                    df_input.columns[int(i)]
                    for i in selected
                ]

        else:

            selected = (
                [selected_columns]
                if selected_columns in df_input.columns
                else []
            )

        steps = getattr(
            transformer,
            "steps",
            []
        )

        step_names = " ".join(
            str(name).lower()
            for name, _ in steps
        )

        transformer_text = (
            str(transformer_name).lower()
            + " "
            + transformer.__class__.__name__.lower()
            + " "
            + step_names
        )

        has_numeric_imputer = any(
            getattr(step, "strategy", None)
            in ("mean", "median")
            for _, step in steps
        )

        if (
            any(
                token in transformer_text
                for token in (
                    "numeric",
                    "num",
                    "scal",
                    "standard"
                )
            )
            or has_numeric_imputer
        ):

            numeric_model_columns.update(selected)

    categorical_columns = {
        "Country",
        "Agrofor_practice",
        "curr_EcC_Transplanting",
        "curr_EcC_total_max",
        "curr_EcB_income_max",
        "curr_EcB_tree"
    }

    # Convert all features into consistent data types.
    for col in df_input.columns:

        series = df_input[col].replace(
            [
                np.inf,
                -np.inf,
                "inf",
                "-inf",
                "Infinity",
                "-Infinity"
            ],
            np.nan
        )

        must_be_numeric = (
            col in numeric_model_columns
            or (
                col in df_reference.columns
                and pd.api.types.is_numeric_dtype(
                    df_reference[col]
                )
            )
        )

        if must_be_numeric:

            converted = pd.to_numeric(
                series,
                errors="coerce"
            )

            df_input[col] = (
                converted
                .replace(
                    [np.inf, -np.inf],
                    np.nan
                )
                .fillna(0.0)
                .astype(float)
            )

        elif col in categorical_columns:

            df_input[col] = (
                series
                .fillna("Unspecified")
                .astype(str)
            )

        elif not pd.api.types.is_numeric_dtype(series):

            # Preserve non-numeric features as categorical text.
            df_input[col] = (
                series
                .fillna("Unspecified")
                .astype(str)
            )

        else:

            df_input[col] = (
                pd.to_numeric(
                    series,
                    errors="coerce"
                )
                .fillna(0.0)
            )

    # Final numeric validation.
    for col in numeric_model_columns:

        if col in df_input.columns:

            df_input[col] = pd.to_numeric(
                df_input[col],
                errors="coerce"
            ).replace(
                [np.inf, -np.inf],
                np.nan
            ).fillna(0.0).astype(float)

    return df_input


# ============================================================
# 7. SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.title("Navigation")

    page = st.radio(
        "Go to:",
        [
            "📋 Data Overview",
            "📊 Visualizations",
            "🔮 Model Prediction"
        ]
    )

    st.markdown("---")

    st.subheader("📁 Model Artifacts")

    st.markdown("""
    - **Model File:** `rf_agroforestry_model.joblib`
    - **Preprocessor:** `preprocessor.joblib`
    - **Feature Selection:** `l1_feature_selector.pkl`
    - **R-squared Score:** `~0.7787`
    """)


# ============================================================
# 8. PAGE 1: DATA OVERVIEW
# ============================================================

if page == "📋 Data Overview":

    st.title("🌳 Agroforestry Benefits Prediction")

    st.subheader(
        "Project Overview & Data Insights"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">
                    Total Observations
                </div>
                <div class="metric-value">
                    {len(df_clean)}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">
                    Unique Countries
                </div>
                <div class="metric-value">
                    {df_clean["Country"].nunique()}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">
                    Median Benefits (USD 2020)
                </div>
                <div class="metric-value">
                    ${df_clean["benefitsusd2020"].median():,.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### Dataset Preview")

    st.dataframe(
        df_clean.head(50),
        use_container_width=True
    )


# ============================================================
# 9. PAGE 2: VISUALIZATIONS
# ============================================================

elif page == "📊 Visualizations":

    st.title("📈 Interactive Data Visualizations")

    tab1, tab2 = st.tabs(
        [
            "Country Distributions",
            "Feature Relationships"
        ]
    )

    with tab1:

        st.subheader(
            "Count of Entries per Country"
        )

        fig_country = px.histogram(
            df_clean,
            y="Country",
            color="Country",
            title="Dataset Volume by Country",
            height=600
        )

        fig_country.update_layout(
            showlegend=False,
            title_x=0.5
        )

        st.plotly_chart(
            fig_country,
            use_container_width=True
        )

    with tab2:

        st.subheader(
            "Benefit Distribution (Target Variable)"
        )

        fig_dist = px.histogram(
            df_clean,
            x="benefitsusd2020",
            nbins=30,
            marginal="rug",
            title=(
                "Distribution of Agroforestry Benefits "
                "(USD 2020)"
            ),
            color_discrete_sequence=["#2e6f40"]
        )

        fig_dist.update_layout(
            title_x=0.5
        )

        st.plotly_chart(
            fig_dist,
            use_container_width=True
        )


# ============================================================
# 10. PAGE 3: MODEL PREDICTION
# ============================================================

elif page == "🔮 Model Prediction":

    st.title("🔮 Forecast Economic Benefits")

    st.write(
        "Enter predictive variables below to estimate "
        "predicted agroforestry benefits in USD (2020)."
    )

    with st.form("prediction_form"):

        st.write("#### Input Features")

        col_left, col_right = st.columns(2)

        with col_left:

            totalusd = st.number_input(
                "Total USD 2020 (totalusd2020)",
                min_value=0.0,
                value=152.0
            )

            ecc_total_max = st.number_input(
                "Max Economic Cost (EcC_total_max)",
                min_value=0.0,
                value=0.0
            )

            ecb_income_max = st.number_input(
                "Max Economic Benefit Income (EcB_income_max)",
                min_value=0.0,
                value=0.0
            )

        with col_right:

            ecb_tree = st.number_input(
                "Tree Benefits Value (EcB_tree)",
                min_value=0.0,
                value=0.0
            )

            curr_transplanting = st.selectbox(
                "Transplanting Currency Context",
                [
                    "N/A",
                    "GH₵ ha-1",
                    "FCFA ha-1",
                    "US$ ha-1"
                ]
            )

            curr_ecc_total = st.selectbox(
                "Max Cost Currency Context",
                [
                    "N/A",
                    "GH₵ ha-1",
                    "FCFA ha-1",
                    "US$ ha-1"
                ]
            )

        submitted = st.form_submit_button(
            "Predict"
        )

    if submitted:

        try:

            # Start with a reference observation to preserve
            # the original dataset feature structure.
            df_input = (
                df_clean
                .drop(
                    columns=["benefitsusd2020"],
                    errors="ignore"
                )
                .iloc[[0]]
                .copy()
            )

            # Insert user-provided numerical values.
            df_input["totalusd2020"] = float(totalusd)

            df_input["EcC_total_max"] = float(
                ecc_total_max
            )

            df_input["EcB_income_max"] = float(
                ecb_income_max
            )

            df_input["EcB_tree"] = float(
                ecb_tree
            )

            # Insert categorical currency values.
            if "curr_EcC_Transplanting" in df_input.columns:

                df_input["curr_EcC_Transplanting"] = str(
                    curr_transplanting
                )

            if "curr_EcC_total_max" in df_input.columns:

                df_input["curr_EcC_total_max"] = str(
                    curr_ecc_total
                )

            # Prepare input for the trained preprocessor.
            df_input = prepare_prediction_input(
                df_input,
                df_clean,
                preprocessor
            )

            st.write("### Input validation")

            with st.expander(
                "View processed prediction input"
            ):

                st.dataframe(
                    df_input,
                    use_container_width=True
                )

                st.write(
                    df_input.dtypes.astype(str)
                )

            # Execute preprocessing.
            input_processed = preprocessor.transform(
                df_input
            )

            # Apply trained feature selector.
            if l1_feature_selector is not None:

                input_processed = (
                    l1_feature_selector.transform(
                        input_processed
                    )
                )

            # Predict economic benefits.
            prediction = rf_model.predict(
                input_processed
            )[0]

            st.success(
                f"### 🎉 Predicted Economic Benefits: "
                f"${prediction:,.2f} USD (2020)"
            )

        except Exception as ex:

            st.error(
                "Error executing model pipeline preprocessing: "
                f"{ex}"
            )

            with st.expander(
                "Technical diagnostics"
            ):

                st.write(
                    "Input feature data types:"
                )

                try:

                    st.dataframe(
                        pd.DataFrame({
                            "feature": df_input.columns,
                            "dtype": [
                                str(dtype)
                                for dtype in df_input.dtypes
                            ],
                            "sample_value": [
                                str(df_input.iloc[0][col])
                                for col in df_input.columns
                            ]
                        }),
                        use_container_width=True
                    )

                except Exception:

                    st.caption(
                        "Input diagnostics are unavailable."
                    )

                st.caption(
                    "If the error persists, the model artifacts "
                    "may have been trained using a different "
                    "feature schema or scikit-learn version."
                )


# ============================================================
# 11. FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Agroforestry Benefits Analytics Dashboard
        • Built with Streamlit, Pandas & Scikit-Learn
    </div>
    """,
    unsafe_allow_html=True
)
