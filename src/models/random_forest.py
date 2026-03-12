"""
Random Forest Classifier for High-Risk Pregnancy Prediction

Ensemble model using sklearn RandomForestClassifier.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import joblib


def build_model(class_weight=None):
    """Create a Random Forest Classifier."""
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight=class_weight,
        n_jobs=-1,
    )
    return model


def tune_hyperparameters(X_train, y_train, cv=5, n_iter=50):
    """Perform hyperparameter tuning with RandomizedSearchCV."""
    param_distributions = {
        "n_estimators": [50, 100, 200, 300, 500],
        "max_depth": [5, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False],
    }

    random_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
        random_state=42,
    )
    random_search.fit(X_train, y_train)

    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best F1 score: {random_search.best_score_:.4f}")

    return random_search.best_estimator_, random_search.best_params_


def get_feature_importance(model, feature_names):
    """Extract feature importance from random forest."""
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    return importance_df


def save_model(model, path):
    """Save trained model to disk."""
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path):
    """Load trained model from disk."""
    return joblib.load(path)
