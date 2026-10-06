import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def identify_features(df, target_column):
    # Separate numerical and categorical features.
    X = df.drop(columns=[target_column])
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