"""
Data Preprocessing Module

Handles feature engineering, encoding, scaling, train/test splitting,
and class imbalance for the maternal health dataset.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import os


def load_dataset(path):
    """Load the synthetic dataset from CSV."""
    df = pd.read_csv(path)
    print(f"Loaded dataset: {df.shape[0]} records, {df.shape[1]} columns")
    return df


def encode_categorical(df):
    """Encode categorical variables using one-hot encoding for nominal features."""
    df = df.copy()
    encoders = {}

    # One-hot encode facility_type (nominal — no ordinal relationship)
    if "facility_type" in df.columns:
        dummies = pd.get_dummies(df["facility_type"], prefix="facility", drop_first=True)
        df = pd.concat([df.drop(columns=["facility_type"]), dummies], axis=1)
        encoders["facility_type"] = list(dummies.columns)

    # Ordinal encode socioeconomic_status (Low=0, Middle=1, High=2)
    if "socioeconomic_status" in df.columns:
        ses_map = {"Low": 0, "Middle": 1, "High": 2}
        df["socioeconomic_status"] = df["socioeconomic_status"].map(ses_map)
        encoders["ses_map"] = ses_map

    return df, encoders


def get_feature_target_split(df, target_col="risk_label", exclude_cols=None):
    """
    Split DataFrame into features (X) and target (y).

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
    exclude_cols : list of str
        Columns to exclude from features (e.g., patient_id).

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    if exclude_cols is None:
        exclude_cols = ["patient_id"]

    feature_cols = [c for c in df.columns if c != target_col and c not in exclude_cols]
    X = df[feature_cols]
    y = df[target_col]

    return X, y


def split_data(X, y, test_size=0.15, val_size=0.15, random_state=42):
    """
    Split data into train, validation, and test sets (70/15/15).

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Second split: separate validation set from remaining
    val_fraction = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_fraction, random_state=random_state, stratify=y_temp
    )

    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    print(f"Train risk prevalence: {y_train.mean():.3f}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test):
    """
    Fit StandardScaler on training data and transform all sets.
    Binary/one-hot columns are excluded from scaling to preserve interpretability.

    Returns
    -------
    X_train_scaled, X_val_scaled, X_test_scaled, scaler
    """
    # Identify binary columns (one-hot encoded + original binary features)
    binary_cols = [c for c in X_train.columns
                   if X_train[c].dropna().isin([0, 1]).all()]
    continuous_cols = [c for c in X_train.columns if c not in binary_cols]

    print(f"Scaling {len(continuous_cols)} continuous features, "
          f"leaving {len(binary_cols)} binary features unscaled")

    scaler = StandardScaler()

    # Scale only continuous columns
    train_cont = pd.DataFrame(
        scaler.fit_transform(X_train[continuous_cols]),
        columns=continuous_cols, index=X_train.index
    )
    val_cont = pd.DataFrame(
        scaler.transform(X_val[continuous_cols]),
        columns=continuous_cols, index=X_val.index
    )
    test_cont = pd.DataFrame(
        scaler.transform(X_test[continuous_cols]),
        columns=continuous_cols, index=X_test.index
    )

    # Reassemble with binary columns unchanged (preserve original index alignment)
    X_train_scaled = pd.concat([train_cont, X_train[binary_cols]], axis=1)
    X_val_scaled = pd.concat([val_cont, X_val[binary_cols]], axis=1)
    X_test_scaled = pd.concat([test_cont, X_test[binary_cols]], axis=1)

    # Restore original column order and reset index for clean downstream use
    X_train_scaled = X_train_scaled[X_train.columns].reset_index(drop=True)
    X_val_scaled = X_val_scaled[X_val.columns].reset_index(drop=True)
    X_test_scaled = X_test_scaled[X_test.columns].reset_index(drop=True)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def handle_imbalance(X_train, y_train, method="smote", random_state=42):
    """
    Handle class imbalance using SMOTE or class weights.

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.Series
    method : str
        'smote' for SMOTE oversampling, 'weights' for returning class weights.

    Returns
    -------
    X_resampled, y_resampled (if SMOTE)
    or class_weight_dict (if weights)
    """
    if method == "smote":
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE: {len(X_resampled)} samples")
        print(f"Class distribution:\n{pd.Series(y_resampled).value_counts().to_string()}")
        return X_resampled, y_resampled

    elif method == "weights":
        counts = y_train.value_counts()
        total = len(y_train)
        class_weights = {cls: total / (len(counts) * count) for cls, count in counts.items()}
        print(f"Class weights: {class_weights}")
        return class_weights

    else:
        raise ValueError(f"Unknown method: {method}")


def save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test, output_dir):
    """Save processed splits to CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        data = X.copy()
        data["risk_label"] = y.values
        path = os.path.join(output_dir, f"{name}.csv")
        data.to_csv(path, index=False)
        print(f"Saved {name} set to {path}")


def save_scaler(scaler, path):
    """Save fitted scaler for later use."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)
    print(f"Scaler saved to {path}")
