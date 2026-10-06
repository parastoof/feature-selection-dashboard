import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import re

ID_PATTERNS = [
    r"^id$",
    r".*_id$",
    r"^id_.*",
    r".*id$",
    r".*_code$",
    r"^code$",
]


def is_id_like_name(column_name):
    """
    Detect ID-like columns based on column name.
    """

    normalized = str(column_name).strip().lower()

    return any(
        re.match(pattern, normalized)
        for pattern in ID_PATTERNS
    )


def analyze_columns(df, high_cardinality_threshold=0.9):
    """
    Analyze columns and identify potential
    ID-like and high-cardinality columns.
    """

    rows = []

    for column in df.columns:

        unique_count = df[column].nunique(
            dropna=False
        )

        unique_ratio = (
            unique_count / len(df)
            if len(df) > 0
            else 0
        )

        id_like = is_id_like_name(column)

        high_cardinality = (
            unique_ratio >= high_cardinality_threshold
        )

        if id_like:
            status = "ID-like"

        elif high_cardinality:
            status = "High Cardinality"

        else:
            status = "Keep"

        rows.append({
            "Column": column,
            "Data Type": str(df[column].dtype),
            "Unique Values": unique_count,
            "Unique %": round(unique_ratio * 100, 2),
            "ID-like": id_like,
            "High Cardinality": high_cardinality,
            "Status": status
        })

    return pd.DataFrame(rows)

def identify_features(X):
    # Separate numerical and categorical features.
    numerical_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()
    return numerical_features, categorical_features

def build_preprocessor(numerical_features, categorical_features, scale_numeric=True):
    """
    Build preprocessing pipeline for numerical
    and categorical features.
    """

    numerical_steps = [("imputer", SimpleImputer(strategy="median"))]

    if scale_numeric:
        numerical_steps.append(("scaler", StandardScaler()))

    numerical_pipeline = Pipeline(steps=numerical_steps)

    categorical_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent")),
               ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

    preprocessor = ColumnTransformer(
                   transformers=[("numerical", numerical_pipeline, numerical_features),
                                 ("categorical", categorical_pipeline, categorical_features)], remainder="drop")

    return preprocessor