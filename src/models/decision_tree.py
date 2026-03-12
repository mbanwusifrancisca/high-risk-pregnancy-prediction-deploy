"""
Decision Tree Classifier for High-Risk Pregnancy Prediction

Interpretable rule-based model using sklearn DecisionTreeClassifier.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import GridSearchCV
import joblib


def build_model(class_weight=None):
    """Create a Decision Tree Classifier."""
    model = DecisionTreeClassifier(
        random_state=42,
        class_weight=class_weight,
    )
    return model


def tune_hyperparameters(X_train, y_train, cv=5):
    """Perform hyperparameter tuning with GridSearchCV."""
    param_grid = {
        "max_depth": [3, 5, 7, 10],
        "min_samples_split": [5, 10, 20, 30],
        "min_samples_leaf": [5, 10, 20],
        "criterion": ["gini", "entropy"],
        "ccp_alpha": [0.0, 0.001, 0.005, 0.01],
    }

    grid_search = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best F1 score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_


def get_feature_importance(model, feature_names):
    """Extract feature importance from the decision tree."""
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    return importance_df


def get_tree_rules(model, feature_names, max_depth=5):
    """Export human-readable decision rules."""
    return export_text(model, feature_names=list(feature_names), max_depth=max_depth)


def save_model(model, path):
    """Save trained model to disk."""
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path):
    """Load trained model from disk."""
    return joblib.load(path)
