import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def split_data(
    X,
    y,
    problem_type,
    test_size=0.2,
    random_state=42
):

    stratify = y if problem_type == "Classification" else None

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )


def build_model(problem_type):

    if problem_type == "Classification":

        return LogisticRegression(
            max_iter=2000
        )

    return LinearRegression()


def evaluate_classification(
    model,
    X_test,
    y_test
):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "Accuracy": accuracy_score(
            y_test,
            predictions
        ),
        "F1": f1_score(
            y_test,
            predictions,
            average="weighted"
        ),
        "ROC-AUC": roc_auc_score(
            y_test,
            probabilities
        )
    }


def evaluate_regression(
    model,
    X_test,
    y_test
):

    predictions = model.predict(X_test)

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    return {
        "MAE": mean_absolute_error(
            y_test,
            predictions
        ),
        "RMSE": rmse,
        "R²": r2_score(
            y_test,
            predictions
        )
    }


def evaluate_model(
    model,
    X_test,
    y_test,
    problem_type
):

    if problem_type == "Classification":

        return evaluate_classification(
            model,
            X_test,
            y_test
        )

    return evaluate_regression(
        model,
        X_test,
        y_test
    )