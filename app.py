
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
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown("""
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
""", unsafe_allow_html=True)


# ============================================================
# 3. FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def find_file(*filenames):

    for filename in filenames:

        path = os.path.join(BASE_DIR, filename)

        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "Could not find any of these files: "
        + ", ".join(filenames)
    )


# ============================================================
# 4. LOAD AND CLEAN DATASET
# ============================================================

@st.cache_data
def load_data():

    csv_path = find_file(
        "Costs_and_benefits_of_agroforestry_in_Africa.csv"
    )

    df = pd.read_csv(
        csv_path,
        encoding="latin1"
    )

    df = df.copy()

    df = df.drop(
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

    for col in df.columns:

        if col not in categorical_columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    if "benefitsusd2020" in df.columns:
        df["benefitsusd2020"] = (
            df["benefitsusd2020"].fillna(99.5)
        )

    if "totalusd2020" in df.columns:
        df["totalusd2020"] = (
            df["totalusd2020"].fillna(152.0)
        )

    for col in categorical_columns:

        if col in df.columns:

            df[col] = (
                df[col]
                .fillna("Unspecified")
                .astype(str)
            )

    for col in [
        "EcC_total_max",
        "EcB_income_max",
        "EcB_tree"
    ]:

        if col in df.columns:

            df[col] = (
                df[col]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )

    return df


# ============================================================
# 5. LOAD MODEL ARTIFACTS
# ============================================================

try:

    df_clean = load_data()

    rf_model = joblib.load(
        find_file(
            "rf_agroforestry_model.joblib",
            "rf_agroforestry_model.pkl"
        )
    )

    preprocessor = joblib.load(
        find_file(
            "preprocessor.joblib",
            "preprocessor.pkl"
        )
    )

    l1_feature_selector = joblib.load(
        find_file(
            "l1_feature_selector.pkl",
            "l1_feature_selector.joblib"
        )
    )

except Exception as e:

    st.error("Unable to load the dataset or model files.")

    st.exception(e)

    st.stop()


# ============================================================
# 6. PREPROCESSOR DIAGNOSTIC FUNCTIONS
# ============================================================

def inspect_transformer(transformer, name="Preprocessor"):

    results = []

    if isinstance(transformer, OneHotEncoder):

        encoder_info = {
            "Transformer": name,
            "Type": "OneHotEncoder",
            "Categories": []
        }

        if hasattr(transformer, "categories_"):

            for i, categories in enumerate(
                transformer.categories_
            ):

                category_types = sorted(
                    set(
                        type(value).__name__
                        for value in categories
                    )
                )

                encoder_info["Categories"].append({
                    "Group": i,
                    "Data type": str(categories.dtype),
                    "Python types": ", ".join(category_types),
                    "Values": [
                        str(value)
                        for value in categories[:30]
                    ],
                    "Number of categories": len(categories)
                })

        results.append(encoder_info)

    elif isinstance(transformer, Pipeline):

        for step_name, step in transformer.steps:

            results.extend(
                inspect_transformer(
                    step,
                    f"{name} -> {step_name}"
                )
            )

    elif isinstance(transformer, ColumnTransformer):

        for step_name, step, columns in transformer.transformers_:

            if step == "drop" or step == "passthrough":
                continue

            results.extend(
                inspect_transformer(
                    step,
                    f"{name} -> {step_name} | columns={columns}"
                )
            )

    return results


def show_model_diagnostics():

    st.subheader("Model and Library Versions")

    st.write("Scikit-learn:", sklearn.__version__)
    st.write("NumPy:", np.__version__)
    st.write("Pandas:", pd.__version__)

    st.subheader("Preprocessor Information")

    st.write(
        "Preprocessor class:",
        type(preprocessor).__name__
    )

    st.write(
        "Expected input features:",
        getattr(
            preprocessor,
            "feature_names_in_",
            "Not available"
        )
    )

    encoder_results = inspect_transformer(
        preprocessor
    )

    if encoder_results:

        for encoder in encoder_results:

            st.markdown(
                f"**{encoder['Transformer']}**"
            )

            for category in encoder["Categories"]:

                st.write(
                    f"Category group {category['Group']}"
                )

                st.write(
                    "Array data type:",
                    category["Data type"]
                )

                st.write(
                    "Python value types:",
                    category["Python types"]
                )

                st.write(
                    "Number of categories:",
                    category["Number of categories"]
                )

                st.write(
                    "First 30 category values:",
                    category["Values"]
                )

    else:

        st.warning(
            "No fitted OneHotEncoder was found in the "
            "preprocessor structure."
        )


# ============================================================
# 7. INPUT PREPARATION
# ============================================================

CATEGORICAL_COLUMNS = {
    "Country",
    "Agrofor_practice",
    "curr_EcC_Transplanting",
    "curr_EcC_total_max",
    "curr_EcB_income_max",
    "curr_EcB_tree"
}


def get_expected_columns():

    expected = getattr(
        preprocessor,
        "feature_names_in_",
        None
    )

    if expected is not None:
        return list(expected)

    return None


def get_numeric_columns():

    numeric_columns = set()

    transformers = getattr(
        preprocessor,
        "transformers_",
        []
    )

    for name, transformer, columns in transformers:

        if name == "remainder":
            continue

        if isinstance(transformer, str):
            continue

        if isinstance(columns, slice):
            continue

        if isinstance(columns, str):
            selected_columns = [columns]

        elif isinstance(
            columns,
            (list, tuple, np.ndarray, pd.Index)
        ):

            selected_columns = list(columns)

            if selected_columns and isinstance(
                selected_columns[0],
                (bool, np.bool_, int, np.integer)
            ):
                continue

        else:
            continue

        steps = getattr(
            transformer,
            "steps",
            []
        )

        description = (
            str(name).lower()
            + " "
            + transformer.__class__.__name__.lower()
            + " "
            + " ".join(
                str(step_name).lower()
                for step_name, _ in steps
            )
        )

        numeric_imputer = any(
            getattr(step, "strategy", None)
            in ("mean", "median")
            for _, step in steps
        )

        if (
            any(
                keyword in description
                for keyword in [
                    "numeric",
                    "num",
                    "scal",
                    "standard"
                ]
            )
            or numeric_imputer
        ):

            numeric_columns.update(
                selected_columns
            )

    return numeric_columns


def prepare_prediction_input(
    df_input,
    df_reference
):

    df_input = df_input.copy()

    expected_columns = get_expected_columns()

    if expected_columns is not None:

        missing = [
            col for col in expected_columns
            if col not in df_input.columns
        ]

        if missing:

            raise ValueError(
                "Missing required model features: "
                + str(missing)
            )

        df_input = df_input.reindex(
            columns=expected_columns
        )

    numeric_columns = get_numeric_columns()

    for col in df_input.columns:

        if (
            col in df_reference.columns
            and pd.api.types.is_numeric_dtype(
                df_reference[col]
            )
        ):
            numeric_columns.add(col)

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

        if col in numeric_columns:

            df_input[col] = (
                pd.to_numeric(
                    series,
                    errors="coerce"
                )
                .replace(
                    [np.inf, -np.inf],
                    np.nan
                )
                .fillna(0.0)
                .astype(np.float64)
            )

        elif col in CATEGORICAL_COLUMNS:

            df_input[col] = (
                series
                .fillna("Unspecified")
                .astype(str)
            )

        elif pd.api.types.is_numeric_dtype(series):

            df_input[col] = (
                pd.to_numeric(
                    series,
                    errors="coerce"
                )
                .replace(
                    [np.inf, -np.inf],
                    np.nan
                )
                .fillna(0.0)
                .astype(np.float64)
            )

        else:

            df_input[col] = (
                series
                .fillna("Unspecified")
                .astype(str)
            )

    return df_input


# ============================================================
# 8. SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🌳 Agroforestry Analytics")

    page = st.radio(
        "Navigate Dashboard",
        [
            "📋 Data Overview",
            "📊 Visualizations",
            "🔮 Model Prediction"
        ]
    )

    st.markdown("---")

    st.subheader("Model Information")

    st.markdown("""
    **Prediction Model:** Random Forest

    **Preprocessing:** ColumnTransformer

    **Feature Selection:** L1 Selector

    **Reported R-squared:** Approximately 0.7787
    """)


# ============================================================
# 9. PAGE: DATA OVERVIEW
# ============================================================

if page == "📋 Data Overview":

    st.title("🌳 Agroforestry Benefits Prediction")

    st.subheader("Project Overview and Data Insights")

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

        country_count = (
            df_clean["Country"].nunique()
            if "Country" in df_clean.columns
            else 0
        )

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">
                    Unique Countries
                </div>
                <div class="metric-value">
                    {country_count}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        median_benefits = (
            df_clean["benefitsusd2020"].median()
            if "benefitsusd2020" in df_clean.columns
            else 0
        )

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">
                    Median Benefits (USD 2020)
                </div>
                <div class="metric-value">
                    ${median_benefits:,.2f}
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

    st.markdown("### Dataset Statistics")

    st.dataframe(
        df_clean.describe(
            include="all"
        ).transpose(),
        use_container_width=True
    )

    st.markdown("### Missing Values")

    missing = (
        df_clean.isnull()
        .sum()
        .reset_index()
    )

    missing.columns = [
        "Feature",
        "Missing Values"
    ]

    st.dataframe(
        missing,
        use_container_width=True
    )


# ============================================================
# 10. PAGE: VISUALIZATIONS
# ============================================================

elif page == "📊 Visualizations":

    st.title("📊 Interactive Data Visualizations")

    tab1, tab2, tab3 = st.tabs(
        [
            "Country Distribution",
            "Benefits Distribution",
            "Feature Relationships"
        ]
    )

    with tab1:

        if "Country" in df_clean.columns:

            country_counts = (
                df_clean["Country"]
                .value_counts()
                .reset_index()
            )

            country_counts.columns = [
                "Country",
                "Observations"
            ]

            fig = px.bar(
                country_counts,
                x="Country",
                y="Observations",
                color="Country",
                title="Dataset Volume by Country"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with tab2:

        if "benefitsusd2020" in df_clean.columns:

            fig = px.histogram(
                df_clean,
                x="benefitsusd2020",
                nbins=30,
                marginal="rug",
                title="Benefits Distribution (USD 2020)",
                color_discrete_sequence=["#2e6f40"]
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.write(
                df_clean["benefitsusd2020"].describe()
            )

    with tab3:

        numeric_df = df_clean.select_dtypes(
            include=np.number
        )

        numeric_columns = list(
            numeric_df.columns
        )

        if len(numeric_columns) >= 2:

            x_feature = st.selectbox(
                "Select X variable",
                numeric_columns
            )

            y_feature = st.selectbox(
                "Select Y variable",
                numeric_columns,
                index=1
            )

            fig = px.scatter(
                df_clean,
                x=x_feature,
                y=y_feature,
                color=(
                    "Country"
                    if "Country" in df_clean.columns
                    else None
                ),
                title=f"{x_feature} vs {y_feature}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.info(
                "Not enough numerical columns "
                "for relationship visualization."
            )


# ============================================================
# 11. PAGE: MODEL PREDICTION
# ============================================================

elif page == "🔮 Model Prediction":

    st.title("🔮 Forecast Agroforestry Economic Benefits")

    st.write(
        "Enter the required information to estimate "
        "agroforestry benefits in USD (2020)."
    )

    # --------------------------------------------------------
    # Prediction form
    # --------------------------------------------------------

    with st.form("prediction_form"):

        st.subheader("Input Features")

        col1, col2 = st.columns(2)

        with col1:

            totalusd = st.number_input(
                "Total USD 2020",
                min_value=0.0,
                value=152.0,
                step=1.0
            )

            ecc_total_max = st.number_input(
                "Maximum Economic Cost",
                min_value=0.0,
                value=0.0,
                step=1.0
            )

            ecb_income_max = st.number_input(
                "Maximum Economic Benefit Income",
                min_value=0.0,
                value=0.0,
                step=1.0
            )

        with col2:

            ecb_tree = st.number_input(
                "Tree Benefits Value",
                min_value=0.0,
                value=0.0,
                step=1.0
            )

            currency_options = [
                "N/A",
                "GH₵ ha-1",
                "FCFA ha-1",
                "US$ ha-1"
            ]

            curr_transplanting = st.selectbox(
                "Transplanting Currency Context",
                currency_options
            )

            curr_ecc_total = st.selectbox(
                "Maximum Cost Currency Context",
                currency_options
            )

        submitted = st.form_submit_button(
            "🔮 Predict Benefits",
            use_container_width=True
        )

    # --------------------------------------------------------
    # Run prediction
    # --------------------------------------------------------

    if submitted:

        df_input = None

        try:

            if df_clean.empty:
                raise ValueError(
                    "Dataset contains no observations."
                )

            if "benefitsusd2020" not in df_clean.columns:
                raise ValueError(
                    "Target column benefitsusd2020 is missing."
                )

            # Use a real observation as a starting template.
            df_input = (
                df_clean
                .drop(
                    columns=["benefitsusd2020"],
                    errors="ignore"
                )
                .iloc[[0]]
                .copy()
            )

            # User-entered values.
            user_values = {
                "totalusd2020": float(totalusd),
                "EcC_total_max": float(ecc_total_max),
                "EcB_income_max": float(ecb_income_max),
                "EcB_tree": float(ecb_tree),
                "curr_EcC_Transplanting": str(
                    curr_transplanting
                ),
                "curr_EcC_total_max": str(
                    curr_ecc_total
                )
            }

            for feature, value in user_values.items():

                if feature in df_input.columns:

                    df_input.loc[
                        df_input.index[0],
                        feature
                    ] = value

            # Prepare input.
            df_input = prepare_prediction_input(
                df_input,
                df_clean
            )

            st.success(
                "Input preparation completed."
            )

            with st.expander(
                "View Model Input and Data Types"
            ):

                st.dataframe(
                    df_input,
                    use_container_width=True
                )

                diagnostics = pd.DataFrame({
                    "Feature": df_input.columns,
                    "Data Type": [
                        str(dtype)
                        for dtype in df_input.dtypes
                    ],
                    "Value": [
                        str(df_input.iloc[0][col])
                        for col in df_input.columns
                    ]
                })

                st.dataframe(
                    diagnostics,
                    use_container_width=True
                )

            # -----------------------------------------------
            # PREPROCESSING
            # -----------------------------------------------

            st.write("### Step 1: Preprocessing")

            input_processed = preprocessor.transform(
                df_input
            )

            # -----------------------------------------------
            # FEATURE SELECTION
            # -----------------------------------------------

            st.write("### Step 2: Feature Selection")

            input_processed = (
                l1_feature_selector.transform(
                    input_processed
                )
            )

            # -----------------------------------------------
            # RANDOM FOREST
            # -----------------------------------------------

            st.write("### Step 3: Random Forest Prediction")

            prediction = rf_model.predict(
                input_processed
            )[0]

            # -----------------------------------------------
            # DISPLAY RESULT
            # -----------------------------------------------

            st.success(
                f"🎉 Predicted Economic Benefits: "
                f"${prediction:,.2f} USD (2020)"
            )

            st.metric(
                "Predicted Agroforestry Benefits",
                f"${prediction:,.2f}"
            )

            st.caption(
                "Prediction generated using the loaded "
                "preprocessor, feature selector, and "
                "Random Forest model."
            )

        except Exception as ex:

            st.error(
                "The prediction pipeline encountered an error."
            )

            st.error(str(ex))

            # Full traceback.
            with st.expander(
                "Full Technical Error and Traceback",
                expanded=True
            ):

                st.code(
                    traceback.format_exc(),
                    language="python"
                )

            # Input diagnostics.
            with st.expander(
                "Input Data Diagnostics",
                expanded=True
            ):

                if df_input is not None:

                    st.write("Input feature data types:")

                    st.dataframe(
                        pd.DataFrame({
                            "Feature": df_input.columns,
                            "Data Type": [
                                str(dtype)
                                for dtype in df_input.dtypes
                            ],
                            "Sample Value": [
                                str(df_input.iloc[0][col])
                                for col in df_input.columns
                            ]
                        }),
                        use_container_width=True
                    )

            # Encoder diagnostics.
            with st.expander(
                "OneHotEncoder and Model Compatibility Diagnostics",
                expanded=True
            ):

                show_model_diagnostics()


# ============================================================
# 12. FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Agroforestry Benefits Analytics Dashboard
        <br>
        Built with Streamlit, Pandas, NumPy,
        Plotly and Scikit-Learn
    </div>
    """,
    unsafe_allow_html=True
)
