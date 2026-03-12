"""
Full ML Pipeline: Preprocess → Train → Evaluate → Compare

Trains four models on the synthetic maternal health dataset:
  1. Logistic Regression (baseline, interpretable)
  2. Decision Tree (rule-based, interpretable)
  3. Random Forest (ensemble)
  4. LSTM Neural Network (deep learning)

Outputs:
  - Trained model files in models/
  - Evaluation metrics, confusion matrices, ROC curves in reports/
  - Comparison table across all models
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data.preprocess import (
    load_dataset, encode_categorical, get_feature_target_split,
    split_data, scale_features, handle_imbalance,
    save_processed_data, save_scaler,
)
from src.models import logistic_regression as lr_module
from src.models import decision_tree as dt_module
from src.models import random_forest as rf_module
from src.models import lstm_model as mlp_module
from src.evaluation.metrics import (
    compute_metrics, find_optimal_threshold, print_classification_report,
    plot_confusion_matrix, plot_all_roc_curves, create_comparison_table,
)

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "synthetic", "maternal_health_synthetic.csv")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

for d in [PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: DATA PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
def run_preprocessing():
    print("\n" + "=" * 70)
    print("PHASE 1: DATA PREPROCESSING")
    print("=" * 70)

    df = load_dataset(DATA_PATH)
    print(f"\nTarget distribution:\n{df['risk_label'].value_counts(normalize=True).round(3)}")

    # Encode categoricals
    df_encoded, encoders = encode_categorical(df)
    print(f"\nAfter encoding: {df_encoded.shape[1]} columns")

    # Feature / target split
    X, y = get_feature_target_split(df_encoded)
    print(f"Features: {list(X.columns)}")
    print(f"Feature count: {X.shape[1]}")

    # Train / Val / Test split (70/15/15)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Scale features
    X_train_sc, X_val_sc, X_test_sc, scaler = scale_features(X_train, X_val, X_test)

    # Save processed data and scaler
    save_processed_data(X_train_sc, X_val_sc, X_test_sc, y_train, y_val, y_test, PROCESSED_DIR)
    save_scaler(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))

    # Handle class imbalance with SMOTE for sklearn models
    X_train_smote, y_train_smote = handle_imbalance(X_train_sc, y_train, method="smote")

    # Also compute class weights for models that support it
    class_weights = handle_imbalance(X_train_sc, y_train, method="weights")

    return (X_train_sc, X_val_sc, X_test_sc, y_train, y_val, y_test,
            X_train_smote, y_train_smote, class_weights, X.columns.tolist())


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════════
def train_logistic_regression(X_train, y_train, X_val, y_val, feature_names):
    print("\n" + "-" * 50)
    print("Training: Logistic Regression")
    print("-" * 50)

    best_model, best_params = lr_module.tune_hyperparameters(X_train, y_train, cv=5)
    y_val_pred = best_model.predict(X_val)
    y_val_prob = best_model.predict_proba(X_val)[:, 1]
    val_metrics = compute_metrics(y_val, y_val_pred, y_val_prob)
    print(f"Validation F1: {val_metrics['f1_score']:.4f}, AUC: {val_metrics['roc_auc']:.4f}")

    importance = lr_module.get_feature_importance(best_model, feature_names)
    print(f"\nTop 5 features:\n{importance.head().to_string(index=False)}")

    lr_module.save_model(best_model, os.path.join(MODELS_DIR, "logistic_regression.joblib"))
    return best_model, best_params


def train_decision_tree(X_train, y_train, X_val, y_val, feature_names):
    print("\n" + "-" * 50)
    print("Training: Decision Tree")
    print("-" * 50)

    best_model, best_params = dt_module.tune_hyperparameters(X_train, y_train, cv=5)
    y_val_pred = best_model.predict(X_val)
    y_val_prob = best_model.predict_proba(X_val)[:, 1]
    val_metrics = compute_metrics(y_val, y_val_pred, y_val_prob)
    print(f"Validation F1: {val_metrics['f1_score']:.4f}, AUC: {val_metrics['roc_auc']:.4f}")

    importance = dt_module.get_feature_importance(best_model, feature_names)
    print(f"\nTop 5 features:\n{importance.head().to_string(index=False)}")

    rules = dt_module.get_tree_rules(best_model, feature_names, max_depth=3)
    rules_path = os.path.join(REPORTS_DIR, "decision_tree_rules.txt")
    with open(rules_path, "w") as f:
        f.write(rules)
    print(f"Decision rules saved to {rules_path}")

    dt_module.save_model(best_model, os.path.join(MODELS_DIR, "decision_tree.joblib"))
    return best_model, best_params


def train_random_forest(X_train, y_train, X_val, y_val, feature_names):
    print("\n" + "-" * 50)
    print("Training: Random Forest")
    print("-" * 50)

    best_model, best_params = rf_module.tune_hyperparameters(
        X_train, y_train, cv=5, n_iter=30
    )
    y_val_pred = best_model.predict(X_val)
    y_val_prob = best_model.predict_proba(X_val)[:, 1]
    val_metrics = compute_metrics(y_val, y_val_pred, y_val_prob)
    print(f"Validation F1: {val_metrics['f1_score']:.4f}, AUC: {val_metrics['roc_auc']:.4f}")

    importance = rf_module.get_feature_importance(best_model, feature_names)
    print(f"\nTop 5 features:\n{importance.head().to_string(index=False)}")

    rf_module.save_model(best_model, os.path.join(MODELS_DIR, "random_forest.joblib"))
    return best_model, best_params


def train_mlp(X_train, y_train, X_val, y_val, class_weights):
    print("\n" + "-" * 50)
    print("Training: Deep Neural Network (MLP)")
    print("-" * 50)

    X_train_np = mlp_module.prepare_input(X_train)
    X_val_np = mlp_module.prepare_input(X_val)
    n_features = X_train_np.shape[1]

    model = mlp_module.build_model(n_features, learning_rate=0.001)
    model.summary()

    history = mlp_module.train_model(
        model, X_train_np, y_train.values,
        X_val_np, y_val.values,
        epochs=100, batch_size=32, class_weight=class_weights,
    )

    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("MLP Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["accuracy"], label="Train Acc")
    axes[1].plot(history.history["val_accuracy"], label="Val Acc")
    axes[1].set_title("MLP Training Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "mlp_training_history.png"), dpi=150)
    plt.close(fig)
    print("Training history plot saved.")

    mlp_module.save_model(model, os.path.join(MODELS_DIR, "mlp_model.keras"))
    return model, history


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: EVALUATION ON TEST SET
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate_all(models, X_test, y_test, X_val, y_val, feature_names):
    print("\n" + "=" * 70)
    print("PHASE 3: EVALUATION ON TEST SET (with threshold optimisation)")
    print("=" * 70)

    all_metrics = {}
    results_dict = {}
    optimal_thresholds = {}

    # --- Sklearn models ---
    sklearn_models = {
        "Logistic Regression": models["lr"],
        "Decision Tree": models["dt"],
        "Random Forest": models["rf"],
    }

    for name, model in sklearn_models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        y_val_prob = model.predict_proba(X_val)[:, 1]

        # Find F1-optimal threshold on VALIDATION set, apply to TEST set
        opt_thresh = find_optimal_threshold(y_val, y_val_prob, metric="f1")
        optimal_thresholds[name] = opt_thresh
        y_pred = (y_prob >= opt_thresh).astype(int)
        print(f"\n{name}: optimal threshold = {opt_thresh:.3f}")

        metrics = compute_metrics(y_test, y_pred, y_prob)
        all_metrics[name] = metrics
        results_dict[name] = {"y_pred": y_pred, "y_prob": y_prob}

        print_classification_report(y_test, y_pred, model_name=name)
        fig = plot_confusion_matrix(
            y_test, y_pred, model_name=name,
            save_path=os.path.join(FIGURES_DIR, f"cm_{name.lower().replace(' ', '_')}.png"),
        )
        plt.close(fig)

    # --- MLP model ---
    mlp_model = models["mlp"]
    X_test_np = mlp_module.prepare_input(X_test)
    X_val_np = mlp_module.prepare_input(X_val)
    _, y_prob_mlp = mlp_module.predict(mlp_model, X_test_np)
    _, y_val_prob_mlp = mlp_module.predict(mlp_model, X_val_np)

    opt_thresh_mlp = find_optimal_threshold(y_val, y_val_prob_mlp, metric="f1")
    optimal_thresholds["MLP"] = opt_thresh_mlp
    y_pred_mlp = (y_prob_mlp >= opt_thresh_mlp).astype(int)
    print(f"\nMLP: optimal threshold = {opt_thresh_mlp:.3f}")

    metrics_mlp = compute_metrics(y_test, y_pred_mlp, y_prob_mlp)
    all_metrics["MLP"] = metrics_mlp
    results_dict["MLP"] = {"y_pred": y_pred_mlp, "y_prob": y_prob_mlp}

    print_classification_report(y_test, y_pred_mlp, model_name="MLP")
    fig = plot_confusion_matrix(
        y_test, y_pred_mlp, model_name="MLP",
        save_path=os.path.join(FIGURES_DIR, "cm_mlp.png"),
    )
    plt.close(fig)

    # --- Combined ROC curve ---
    fig = plot_all_roc_curves(
        results_dict, y_test,
        save_path=os.path.join(FIGURES_DIR, "roc_comparison.png"),
    )
    plt.close(fig)

    # --- Comparison table ---
    comparison_df = create_comparison_table(
        all_metrics,
        save_path=os.path.join(REPORTS_DIR, "model_comparison.csv"),
    )

    # --- Feature importance comparison (bar chart) ---
    plot_feature_importance_comparison(models, feature_names)

    # Save all metrics and thresholds as JSON
    with open(os.path.join(REPORTS_DIR, "all_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    with open(os.path.join(REPORTS_DIR, "optimal_thresholds.json"), "w") as f:
        json.dump(optimal_thresholds, f, indent=2)
    print(f"\nOptimal thresholds: {optimal_thresholds}")

    return all_metrics, comparison_df


def plot_feature_importance_comparison(models, feature_names):
    """Plot feature importance for tree-based models side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Logistic Regression coefficients
    lr_imp = lr_module.get_feature_importance(models["lr"], feature_names)
    axes[0].barh(lr_imp["feature"][:10], lr_imp["abs_coefficient"][:10], color="steelblue")
    axes[0].set_title("Logistic Regression\n(|Coefficient|)")
    axes[0].invert_yaxis()

    # Decision Tree importance
    dt_imp = dt_module.get_feature_importance(models["dt"], feature_names)
    axes[1].barh(dt_imp["feature"][:10], dt_imp["importance"][:10], color="darkorange")
    axes[1].set_title("Decision Tree\n(Gini Importance)")
    axes[1].invert_yaxis()

    # Random Forest importance
    rf_imp = rf_module.get_feature_importance(models["rf"], feature_names)
    axes[2].barh(rf_imp["feature"][:10], rf_imp["importance"][:10], color="forestgreen")
    axes[2].set_title("Random Forest\n(Gini Importance)")
    axes[2].invert_yaxis()

    plt.suptitle("Feature Importance Comparison (Top 10)", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "feature_importance_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Feature importance plot saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: STRATIFIED K-FOLD CROSS-VALIDATION WITH CONFIDENCE INTERVALS
# ═══════════════════════════════════════════════════════════════════════════════
def run_kfold_cv(X, y, feature_names, class_weights, k=5):
    """
    Run stratified k-fold CV on sklearn models to produce mean ± std metrics
    with 95% confidence intervals. This addresses the single-split limitation
    and provides statistical rigor for the paper.
    """
    print("\n" + "=" * 70)
    print(f"PHASE 4: STRATIFIED {k}-FOLD CROSS-VALIDATION")
    print("=" * 70)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from imblearn.over_sampling import SMOTE
    from sklearn.preprocessing import StandardScaler

    model_constructors = {
        "Logistic Regression": lambda: LogisticRegression(
            C=1, penalty="l2", solver="lbfgs", max_iter=1000, random_state=42
        ),
        "Decision Tree": lambda: DecisionTreeClassifier(
            criterion="entropy", max_depth=10, min_samples_leaf=2,
            min_samples_split=5, random_state=42
        ),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_leaf=2,
            max_features="sqrt", random_state=42, n_jobs=-1
        ),
    }

    cv_results = {name: {m: [] for m in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]}
                  for name in model_constructors}

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold_idx + 1}/{k} ---")
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]

        # SMOTE on fold training data only
        smote = SMOTE(random_state=42)
        X_fold_smote, y_fold_smote = smote.fit_resample(X_fold_train, y_fold_train)

        for name, constructor in model_constructors.items():
            model = constructor()
            model.fit(X_fold_smote, y_fold_smote)
            y_pred = model.predict(X_fold_val)
            y_prob = model.predict_proba(X_fold_val)[:, 1]
            metrics = compute_metrics(y_fold_val, y_pred, y_prob)
            for m, v in metrics.items():
                cv_results[name][m].append(v)
            print(f"  {name}: F1={metrics['f1_score']:.4f}, AUC={metrics['roc_auc']:.4f}")

    # Compute mean ± std and 95% CI
    print("\n" + "=" * 60)
    print(f"CROSS-VALIDATION SUMMARY ({k}-Fold)")
    print("=" * 60)

    cv_summary = {}
    for name in model_constructors:
        cv_summary[name] = {}
        for m in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
            values = np.array(cv_results[name][m])
            mean = values.mean()
            std = values.std()
            ci_low = mean - 1.96 * std / np.sqrt(k)
            ci_high = mean + 1.96 * std / np.sqrt(k)
            cv_summary[name][m] = f"{mean:.4f} ± {std:.4f} [{ci_low:.4f}, {ci_high:.4f}]"
        print(f"\n{name}:")
        for m, v in cv_summary[name].items():
            print(f"  {m}: {v}")

    # Save CV results
    with open(os.path.join(REPORTS_DIR, "cv_results.json"), "w") as f:
        json.dump({name: {m: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                               "folds": [float(x) for x in v]}
                          for m, v in metrics_dict.items()}
                   for name, metrics_dict in cv_results.items()}, f, indent=2)
    print(f"\nCV results saved to {os.path.join(REPORTS_DIR, 'cv_results.json')}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   HIGH-RISK PREGNANCY PREDICTION — FULL ML PIPELINE            ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # Phase 1
    (X_train_sc, X_val_sc, X_test_sc, y_train, y_val, y_test,
     X_train_smote, y_train_smote, class_weights, feature_names) = run_preprocessing()

    # Phase 2 — Train all models
    print("\n" + "=" * 70)
    print("PHASE 2: MODEL TRAINING")
    print("=" * 70)

    # Sklearn models use SMOTE-balanced data
    lr_model, lr_params = train_logistic_regression(
        X_train_smote, y_train_smote, X_val_sc, y_val, feature_names
    )
    dt_model, dt_params = train_decision_tree(
        X_train_smote, y_train_smote, X_val_sc, y_val, feature_names
    )
    rf_model, rf_params = train_random_forest(
        X_train_smote, y_train_smote, X_val_sc, y_val, feature_names
    )

    # MLP uses class weights (not SMOTE) — works better with neural nets
    mlp_model, mlp_history = train_mlp(
        X_train_sc, y_train, X_val_sc, y_val, class_weights
    )

    # Phase 3 — Evaluate on held-out test set
    models = {"lr": lr_model, "dt": dt_model, "rf": rf_model, "mlp": mlp_model}
    all_metrics, comparison_df = evaluate_all(models, X_test_sc, y_test, X_val_sc, y_val, feature_names)

    # Phase 4 — Stratified k-fold cross-validation for statistical rigor
    run_kfold_cv(X_train_sc, y_train, feature_names, class_weights)

    # Final summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nBest model by F1: {comparison_df['f1_score'].idxmax()} "
          f"(F1={comparison_df['f1_score'].max():.4f})")
    print(f"Best model by AUC: {comparison_df['roc_auc'].idxmax()} "
          f"(AUC={comparison_df['roc_auc'].max():.4f})")
    print(f"\nOutputs saved to:")
    print(f"  Models:  {MODELS_DIR}/")
    print(f"  Reports: {REPORTS_DIR}/")
    print(f"  Figures: {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
