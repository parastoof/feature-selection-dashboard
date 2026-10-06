import numpy as np
import pandas as pd

from sklearn.feature_selection import (mutual_info_classif, mutual_info_regression,
                                       f_classif, f_regression, RFE)
from sklearn.linear_model import (LogisticRegression,Lasso)
from sklearn.ensemble import (RandomForestClassifier,
                              RandomForestRegressor)


def correlation_selection(X, y, threshold=0.1):
    """
    Select numerical features based on
    absolute correlation with the target.
    """
    data = X.copy()
    data["__target__"] = y.values

    correlations = (data.corr(numeric_only=True)["__target__"].drop("__target__")
                    .abs().sort_values(ascending=False))

    selected_features = correlations[correlations >= threshold].index.tolist()

    return {"selected_features": selected_features,
            "scores": correlations}


def mutual_information_selection(X, y, problem_type, k=10, random_state=42):
    """
    Select top-k features using Mutual Information.
    """

    if problem_type == "Classification":
        scores = mutual_info_classif(X, y, random_state=random_state)

    else:
        scores = mutual_info_regression(X, y, random_state=random_state)

    scores = pd.Series(scores, index=X.columns).sort_values(ascending=False)

    selected_features = (scores.head(min(k, len(scores))).index.tolist())

    return {"selected_features": selected_features,
            "scores": scores}


def f_test_selection(X, y, problem_type, k=10):
    """
    Select top-k features using F-test.
    """

    if problem_type == "Classification":
        scores, p_values = f_classif(X, y)

    else:
        scores, p_values = f_regression(X, y)

    scores = pd.Series(scores, index=X.columns).sort_values(ascending=False)

    selected_features = (scores.head(min(k, len(scores))).index.tolist())

    return {"selected_features": selected_features,
            "scores": scores}


def rfe_selection(X, y, problem_type, n_features_to_select=10):
    """
    Select features using Recursive Feature Elimination.
    """

    if problem_type == "Classification":
        estimator = LogisticRegression(max_iter=2000)

    else:
        estimator = RandomForestRegressor(n_estimators=100,
                                          random_state=42,
                                          n_jobs=-1)

    n_features_to_select = min(n_features_to_select, X.shape[1])

    selector = RFE(estimator=estimator,
                   n_features_to_select=n_features_to_select)

    selector.fit(X, y)

    ranking = pd.Series(selector.ranking_, index=X.columns).sort_values()

    selected_features = X.columns[selector.support_].tolist()

    return {"selected_features": selected_features,
            "scores": ranking,
            "selector": selector}

def l1_selection(X, y, problem_type, threshold=1e-5):
    """
    Embedded feature selection using L1 regularization.
    """

    if problem_type == "Classification":
        estimator = LogisticRegression(penalty="l1",
                                       solver="liblinear",
                                       max_iter=2000)

    else:
        estimator = Lasso(alpha=0.01, max_iter=5000)

    estimator.fit(X, y)

    if problem_type == "Classification":
        coefficients = np.abs(estimator.coef_)

        if coefficients.ndim > 1:
            scores = coefficients.mean(axis=0)
        else:
            scores = coefficients

    else:
        scores = np.abs(estimator.coef_)

    scores = pd.Series(scores, index=X.columns).sort_values(ascending=False)

    selected_features = scores[scores > threshold].index.tolist()

    return {"selected_features": selected_features,
            "scores": scores,
            "estimator": estimator}


def tree_importance_selection(X, y, problem_type, k=10):
    """
    Select top-k features using
    Random Forest feature importance.
    """

    if problem_type == "Classification":

        estimator = RandomForestClassifier(n_estimators=100,
                                           random_state=42,
                                           n_jobs=-1)

    else:
        estimator = RandomForestRegressor(n_estimators=100,
                                          random_state=42,
                                          n_jobs=-1)

    estimator.fit(X, y)

    scores = pd.Series(estimator.feature_importances_, 
                       index=X.columns).sort_values(ascending=False)

    selected_features = (scores.head(min(k, len(scores))).index.tolist())

    return {"selected_features": selected_features,
            "scores": scores,
            "estimator": estimator}

def run_feature_selection(
    method,
    X,
    y,
    problem_type,
    k=10,
    correlation_threshold=0.1
):
    """
    Run a selected feature selection method.
    """

    if method == "Correlation":

        return correlation_selection(
            X,
            y,
            threshold=correlation_threshold
        )

    elif method == "Mutual Information":

        return mutual_information_selection(
            X,
            y,
            problem_type=problem_type,
            k=k
        )

    elif method == "F-test":

        return f_test_selection(
            X,
            y,
            problem_type=problem_type,
            k=k
        )

    elif method == "RFE":

        return rfe_selection(
            X,
            y,
            problem_type=problem_type,
            n_features_to_select=k
        )

    elif method == "L1":

        return l1_selection(
            X,
            y,
            problem_type=problem_type
        )

    elif method == "Random Forest":

        return tree_importance_selection(
            X,
            y,
            problem_type=problem_type,
            k=k
        )

    else:

        raise ValueError(
            f"Unknown feature selection method: {method}"
        )