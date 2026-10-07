import matplotlib.pyplot as plt
import pandas as pd


def plot_model_performance(
    evaluation_df,
    problem_type
):
    """
    Plot model performance for all features
    and feature-selection methods.
    """

    if problem_type == "Classification":
        metric = "ROC-AUC"
        ylabel = "ROC-AUC"
    else:
        metric = "R²"
        ylabel = "R²"

    plot_df = evaluation_df[
        ["Method", metric]
    ].copy()

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        plot_df["Method"],
        plot_df[metric]
    )

    ax.set_title("Model Performance Comparison")
    ax.set_xlabel("Method")
    ax.set_ylabel(ylabel)

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    return fig


def plot_consensus(consensus_df):
    """
    Plot feature selection consensus.
    """

    plot_df = consensus_df.sort_values(
        "Consensus %",
        ascending=True
    ).copy()

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.barh(
        plot_df["Feature"],
        plot_df["Consensus %"]
    )

    ax.set_title("Feature Selection Consensus")
    ax.set_xlabel("Consensus %")
    ax.set_ylabel("Feature")

    plt.tight_layout()

    return fig


def plot_feature_count_vs_performance(
    evaluation_df,
    problem_type
):
    """
    Plot the relationship between the number
    of selected features and model performance.
    """

    if problem_type == "Classification":
        metric = "ROC-AUC"
        ylabel = "ROC-AUC"
    else:
        metric = "R²"
        ylabel = "R²"

    plot_df = evaluation_df[
        ["Method", "Number of Features", metric]
    ].copy()

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(
        plot_df["Number of Features"],
        plot_df[metric]
    )

    for _, row in plot_df.iterrows():
        ax.annotate(
            row["Method"],
            (
                row["Number of Features"],
                row[metric]
            ),
            xytext=(5, 5),
            textcoords="offset points"
        )

    ax.set_title(
        "Number of Features vs Model Performance"
    )
    ax.set_xlabel("Number of Features")
    ax.set_ylabel(ylabel)

    plt.tight_layout()

    return fig
