"""
Model Evaluation Metrics Module

Computes classification metrics and generates evaluation visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
)


def find_optimal_threshold(y_true, y_prob, metric="f1"):
    """
    Find the probability threshold that maximises the chosen metric.
    Uses a fine grid search (robust for both smooth and discrete probabilities).

    Parameters
    ----------
    y_true : array-like
    y_prob : array-like
    metric : str
        'f1' (default) or 'youden' (Youden's J = sensitivity + specificity - 1).

    Returns
    -------
    float
        Optimal threshold.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if metric == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        return float(thresholds[best_idx])

    # F1-optimal: scan a fine grid of thresholds
    best_f1, best_thresh = 0.0, 0.5
    for t in np.arange(0.10, 0.90, 0.01):
        preds = (y_prob >= t).astype(int)
        if preds.sum() == 0:
            continue
        f1_val = f1_score(y_true, preds, zero_division=0)
        if f1_val > best_f1:
            best_f1 = f1_val
            best_thresh = t
    return float(best_thresh)


def compute_metrics(y_true, y_pred, y_prob=None):
    """
    Compute all evaluation metrics.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    y_prob : array-like, optional
        Predicted probabilities for the positive class.

    Returns
    -------
    dict
        Dictionary of metric names and values.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }

    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)

    return metrics


def print_classification_report(y_true, y_pred, model_name="Model"):
    """Print a formatted classification report."""
    print(f"\n{'='*50}")
    print(f"Classification Report: {model_name}")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred, target_names=["Low Risk", "High Risk"]))


def plot_confusion_matrix(y_true, y_pred, model_name="Model", save_path=None):
    """Plot a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Low Risk", "High Risk"],
        yticklabels=["Low Risk", "High Risk"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved confusion matrix to {save_path}")

    return fig


def plot_roc_curve(y_true, y_prob, model_name="Model", ax=None, save_path=None):
    """Plot ROC curve for a single model."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.get_figure()

    ax.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved ROC curve to {save_path}")

    return fig, ax


def plot_all_roc_curves(results_dict, y_true, save_path=None):
    """
    Plot ROC curves for all models on a single figure.

    Parameters
    ----------
    results_dict : dict
        {model_name: {"y_prob": array}} for each model.
    y_true : array-like
        True labels.
    save_path : str, optional
        Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    for model_name, results in results_dict.items():
        y_prob = results["y_prob"]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.4f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve Comparison — All Models", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved ROC comparison to {save_path}")

    return fig


def create_comparison_table(all_metrics, save_path=None):
    """
    Create a comparison table of all model metrics.

    Parameters
    ----------
    all_metrics : dict
        {model_name: metrics_dict} for each model.
    save_path : str, optional
        Path to save the table as CSV.

    Returns
    -------
    pd.DataFrame
    """
    df = pd.DataFrame(all_metrics).T
    df.index.name = "Model"
    df = df.round(4)

    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)
    print(df.to_string())

    if save_path:
        df.to_csv(save_path)
        print(f"\nSaved comparison table to {save_path}")

    return df
