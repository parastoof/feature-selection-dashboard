import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.preprocessing import (identify_features, build_preprocessor, 
                                 analyze_columns, build_feature_mapping)
from utils.feature_selection import run_feature_selection, aggregate_selected_features
from utils.evaluation import (
    split_data,
    build_model,
    evaluate_model
)

# Page Configuration
st.set_page_config(page_title="Feature Selection Dashboard", page_icon="📊", layout="wide")

# Header
st.title("Feature Selection Dashboard")
st.markdown("""An interactive dashboard for selecting and comparing
               machine learning features.""")
st.divider()

# 1. Upload Dataset
st.subheader("1. Upload Dataset")
uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])


if uploaded_file is not None:
    # Read Dataset
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format.")
            st.stop()
        st.success(f"Dataset loaded successfully: {uploaded_file.name}")

    except Exception as e:
        st.error(f"Error while reading the file: {e}")
        st.stop()

    # 2. Dataset Overview
    st.subheader("2. Dataset Overview")
    df_info = pd.DataFrame({"Rows": [df.shape[0]],
                            "Columns": [df.shape[1]],
                            "Missing Values": [int(df.isnull().sum().sum())],
                            "Duplicate Rows": [int(df.duplicated().sum())]})
    st.dataframe(df_info, use_container_width=True, hide_index=True)

    # Dataset Preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head(5), use_container_width=True, hide_index=True)

    # Column Information
    st.subheader("Column Information")
    column_info = pd.DataFrame({"Column": df.columns,
                                "Data Type": df.dtypes.astype(str).values,
                                "Missing Values": df.isnull().sum().values,
                                "Missing %": (df.isnull().mean().values * 100).round(2),
                                "Unique Values": [df[column].nunique() for column in df.columns]})
    st.dataframe(column_info, use_container_width=True, hide_index=True)

    # 3. Dataset Configuration
    st.subheader("3. Dataset Configuration")
    col1, col2 = st.columns(2)
    with col1:
        target_column = st.selectbox("Target Column", options=df.columns)
    with col2:
        problem_type = st.selectbox("Problem Type",
                                    options=["Classification", "Regression"])

    # 4. Column Analysis
    st.subheader("4. Column Analysis")
    column_analysis = analyze_columns(df)

    st.dataframe(column_analysis,
                 use_container_width=True,
                 hide_index=True)


    # Detect Potential Columns
    potential_columns = column_analysis[(column_analysis["Status"] != "Keep") & 
                                        (column_analysis["Column"] != target_column)]["Column"].tolist()

    if potential_columns:
        st.warning("Potential ID-like or high-cardinality "
                   "columns were detected.")
    else:
        st.info( "No potential ID-like or high-cardinality "
                "columns were detected.")

    # Feature Exclusion
    st.subheader("5. Feature Exclusion")
    all_feature_columns = [column
                           for column in df.columns
                           if column != target_column]


    manually_removed = st.multiselect("Select columns to exclude from features",
                                      options=all_feature_columns,
                                      default=potential_columns)


    if manually_removed:
        st.write("Excluded columns:", ", ".join(manually_removed))

    else:
        st.write("No columns excluded.")

    # Prepare X and y
    X = df.drop(columns=[target_column] + manually_removed)
    y = df[target_column]

    # --------------------------------------------------
    # 6. Train / Test Split
    # --------------------------------------------------

    st.subheader("6. Train / Test Split")

    test_size = st.slider(
        "Test set size",
        min_value=0.1,
        max_value=0.4,
        value=0.2,
        step=0.05
    )

    stratify = (y if problem_type == "Classification" else None)

    try:

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=42,
                stratify=stratify))

    except Exception as e:
        st.error(f"Train/test split failed: {e}")
        st.stop()


    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training Rows", X_train.shape[0])

    with col2:
        st.metric("Test Rows", X_test.shape[0])


    # 7. Feature Types
    st.subheader("7. Feature Types")
    numerical_features, categorical_features = (identify_features(X_train))

    col1, col2 = st.columns(2)
    with col1:
        st.write("Numerical Features")
        num_df = pd.DataFrame(numerical_features, columns=["Feature"])
        st.dataframe(num_df,
                    use_container_width=True,
                    hide_index=True)

    with col2:
        st.write("Categorical Features")
        cat_df = pd.DataFrame(categorical_features, columns=["Feature"])

        st.dataframe(cat_df,
                     use_container_width=True,
                     hide_index=True)

    # 8. Preprocessing
    st.subheader("8. Preprocessing")
    scale_numeric = st.checkbox("Apply Standard Scaling to numerical features", value=True)

    preprocessor = build_preprocessor(numerical_features,
                                      categorical_features,
                                      scale_numeric=scale_numeric)
    try:
        X_train_processed = (preprocessor.fit_transform(X_train))
        X_test_processed = (preprocessor.transform(X_test))
        
        feature_names = (preprocessor.get_feature_names_out())
        feature_mapping = build_feature_mapping(preprocessor)
        
        X_train_processed = pd.DataFrame(
            X_train_processed,
            columns=feature_names,
            index=X_train.index)

        X_test_processed = pd.DataFrame(
            X_test_processed,
            columns=feature_names,
            index=X_test.index)
        
        st.subheader("Feature Mapping")
        st.dataframe(feature_mapping,
            use_container_width=True,
            hide_index=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Original Features", X_train.shape[1])

        with col2:
            st.metric("Processed Features", X_train_processed.shape[1])

        st.success("Preprocessing pipeline executed successfully.")


        # 9. Feature Selection
        st.subheader("9. Feature Selection")

        if problem_type == "Regression":
            available_methods = ["Correlation", "Mutual Information",
                                 "F-test", "RFE", "L1", "Random Forest"]
        else:
            available_methods = ["Mutual Information", "F-test",
                                 "RFE", "L1", "Random Forest"]

        selected_methods = st.multiselect("Select one or more methods",
                                          options=available_methods,
                                          default=["Mutual Information"])

        # Feature Selection Parameters
        if selected_methods:
            max_features = X_train_processed.shape[1]

            if max_features > 1:
                n_features = st.slider("Number of features to select",
                                       min_value=1,
                                       max_value=max_features,
                                       value=min(10, max_features))
            else:
                n_features = 1

        if "Correlation" in selected_methods:
            correlation_threshold = st.slider("Correlation threshold",
                                              min_value=0.0, max_value=1.0,
                                              value=0.1, step=0.05)
        else:
            correlation_threshold = 0.1

        # Run Feature Selection
        run_selection = st.button("Run Feature Selection", type="primary")

        if run_selection:
            if not selected_methods:
                st.warning("Please select at least one "
                           "feature selection method.")

            else:
                results = {}
                progress = st.progress(0)

                for i, method in enumerate(selected_methods):
                    try:
                        result = run_feature_selection(method=method,
                                                       X=X_train_processed,
                                                       y=y_train,
                                                       problem_type=problem_type,
                                                       k=n_features,
                                                       correlation_threshold=(correlation_threshold))
                        results[method] = result

                    except Exception as e:
                        st.error(f"{method} failed: {e}")

                    progress.progress((i + 1) / len(selected_methods))

                st.session_state["feature_selection_results"] = results
                st.session_state.pop("evaluation_results", None)
                st.success("Feature selection completed successfully.")

        # Feature Selection Results
        if "feature_selection_results" in st.session_state:
            results = st.session_state["feature_selection_results"]
            st.subheader("Feature Selection Results")
            comparison_data = []

            for method, result in results.items():

                selected_features = result[
                    "selected_features"
                ]

                aggregated = aggregate_selected_features(
                    selected_features,
                    feature_mapping
                )

                aggregated = aggregated.sort_values(
                    "Selection %",
                    ascending=False
                )

                for rank, (_, row) in enumerate(
                    aggregated.iterrows(),
                    start=1
                ):

                    comparison_data.append({
                        "Method": method,
                        "Feature": row["Original Feature"],
                        "Selected Encoded": (
                            row["Selected Encoded Features"]
                        ),
                        "Total Encoded": (
                            row["Total Encoded Features"]
                        ),
                        "Selection %": row["Selection %"],
                        "Rank": rank
                    })


            comparison_df = pd.DataFrame(comparison_data)

            st.subheader("Comparison Matrix")

            original_features = feature_mapping[
                "Original Feature"
            ].unique()


            comparison_matrix = pd.DataFrame(
                index=original_features
            )


            for method, result in results.items():

                selected_features = result[
                    "selected_features"
                ]

                aggregated = aggregate_selected_features(
                    selected_features,
                    feature_mapping
                )

                selected_original_features = set(
                    aggregated["Original Feature"]
                )

                comparison_matrix[method] = [
                    "✓"
                    if feature in selected_original_features
                    else ""
                    for feature in comparison_matrix.index
                ]


            comparison_matrix.index.name = "Feature"


            st.dataframe(
                comparison_matrix,
                use_container_width=True
            )
            st.subheader("Feature Selection Consensus")

        consensus_data = []


        for feature in comparison_matrix.index:

            selected_by = 0

            for method in results:

                if comparison_matrix.loc[
                    feature,
                    method
                ] == "✓":

                    selected_by += 1


            consensus_percentage = (
                selected_by /
                len(results) *
                100
            )


            consensus_data.append({
                "Feature": feature,
                "Selected By": selected_by,
                "Total Methods": len(results),
                "Consensus %": round(
                    consensus_percentage,
                    1
                )
            })


        consensus_df = pd.DataFrame(
            consensus_data
        )


        consensus_df = consensus_df.sort_values(
            "Consensus %",
            ascending=False
        )


        st.dataframe(
            consensus_df,
            use_container_width=True,
            hide_index=True
        )

        # 10. Model Evaluation
        st.subheader("10. Model Evaluation")
        st.markdown("""Evaluate the baseline model using all features
                        and compare it with models trained using
                        selected features.""")

        evaluate_models = st.button("Evaluate Feature Selection", type="primary")

        if evaluate_models:
            if "feature_selection_results" not in st.session_state:
                st.warning("Run feature selection first.")

            else:
                results = st.session_state["feature_selection_results"]
                evaluation_results = []

                # Baseline Model
                baseline_model = build_model(problem_type)
                baseline_model.fit(X_train_processed, y_train)
                baseline_metrics = evaluate_model(baseline_model, X_test_processed, y_test, problem_type)

                baseline_row = {"Method": "All Features",
                                "Number of Features": X_train_processed.shape[1]}
                baseline_row.update(baseline_metrics)
                evaluation_results.append(baseline_row)

                # Evaluate Selected Features
                for method, result in results.items():
                    selected_features = result["selected_features"]

                    X_train_selected = X_train_processed[selected_features]
                    X_test_selected = X_test_processed[selected_features]

                    model = build_model(problem_type)
                    model.fit(X_train_selected, y_train)

                    metrics = evaluate_model(model, X_test_selected,
                                             y_test, problem_type)

                    row = {"Method": method,
                           "Number of Features": len(selected_features)}

                    row.update(metrics)
                    evaluation_results.append(row)

                evaluation_df = pd.DataFrame(evaluation_results)
                st.session_state["evaluation_results"] = evaluation_df

        # Display Evaluation Results
        if "evaluation_results" in st.session_state:
            evaluation_df = st.session_state["evaluation_results"]
            st.dataframe(evaluation_df,
                         use_container_width=True,
                         hide_index=True)
            
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")

else:
    st.info("Upload a CSV or Excel file to get started.")