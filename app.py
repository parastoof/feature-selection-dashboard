import streamlit as st
import pandas as pd
from utils.preprocessing import (identify_features, build_preprocessor)

st.set_page_config(
    page_title="Feature Selection Dashboard",
    page_icon="📊",
    layout="wide"
)


st.title("Feature Selection Dashboard")

st.markdown(
    """
    An interactive dashboard for selecting and comparing
    machine learning features.
    """
)

st.divider()


# --------------------------------------------------
# Upload Dataset
# --------------------------------------------------

st.subheader("1. Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx"]
)


if uploaded_file is not None:

    # Read Dataset
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        st.success(f"Dataset loaded successfully: {uploaded_file.name}")

    except Exception as e:
        st.error(f"Error while reading the file: {e}")
        st.stop()

    # Dataset Overview
    st.subheader("2. Dataset Overview")
    df_info = pd.DataFrame({"Rows": [df.shape[0]],
                            "Columns": [df.shape[1]],
                            "Missing Values": [int(df.isnull().sum().sum())],
                            "Duplicate Rows": [int(df.duplicated().sum())]})
    st.dataframe(df_info, use_container_width=True, hide_index=True)
    
    # Preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head(5), use_container_width=True)

    # Column Information
    st.subheader("Column Information")
    column_info = pd.DataFrame({"Column": df.columns,
                                "Data Type": df.dtypes.astype(str).values,
                                "Missing Values": df.isnull().sum().values,
                                "Missing %": (df.isnull().mean().values * 100).round(2),
                                "Unique Values": [df[column].nunique() for column in df.columns]})
    st.dataframe(column_info, use_container_width=True, hide_index=True)

    # Target & Problem Type
    st.subheader("3. Dataset Configuration")
    col1, col2 = st.columns(2)
    with col1:
        target_column = st.selectbox("Target Column", options=df.columns)
    with col2:
        problem_type = st.selectbox("Problem Type", options=["Classification", "Regression"])

    st.write("Selected Target:", target_column)
    st.write("Problem Type:", problem_type)

    # Feature Detection
    numerical_features, categorical_features = identify_features(
        df,
        target_column
    )

    st.subheader("4. Feature Types")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Numerical Features")
        num_df = pd.DataFrame(numerical_features, columns=["Feature"])
        st.dataframe(num_df, use_container_width=True, hide_index=True)

    with col2:
        st.write("Categorical Features")
        cat_df = pd.DataFrame(categorical_features, columns=["Feature"])
        st.dataframe(cat_df, use_container_width=True, hide_index=True)

    scale_numeric = st.checkbox(
        "Apply Standard Scaling to numerical features",
        value=True
    )
    preprocessor = build_preprocessor(
        numerical_features,
        categorical_features,
        scale_numeric=scale_numeric
    )   

    # Test Preprocessing
# --------------------------------------------------

    X = df.drop(columns=[target_column])
    y = df[target_column]

    try:

        X_processed = preprocessor.fit_transform(X)

        st.subheader("5. Preprocessing Result")

        st.write(
            "Original feature count:",
            X.shape[1]
        )

        st.write(
            "Processed feature count:",
            X_processed.shape[1]
        )

        st.success(
            "Preprocessing pipeline executed successfully."
        )

    except Exception as e:

        st.error(
            f"Preprocessing failed: {e}"
        )





else:
    st.info("Upload a CSV or Excel file to get started.")