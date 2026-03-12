"""
Logistic Regression Model for High-Risk Pregnancy Prediction

Interpretable baseline model using sklearn LogisticRegression.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import joblib


def build_model(class_weight=None):
    """Create a Logistic Regression model."""
    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        class_weight=class_weight,
        random_state=42,
    )
    return model


def tune_hyperparameters(X_train, y_train, cv=5):
    """Perform hyperparameter tuning with GridSearchCV."""
    param_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "penalty": ["l2"],
        "solver": ["lbfgs"],
    }

    grid_search = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42),
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
    """Extract feature importance from logistic regression coefficients."""
    coefficients = model.coef_[0]
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
        "abs_coefficient": np.abs(coefficients),
    }).sort_values("abs_coefficient", ascending=False)

    return importance_df


def save_model(model, path):
    """Save trained model to disk."""
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path):
    """Load trained model from disk."""
    return joblib.load(path)
