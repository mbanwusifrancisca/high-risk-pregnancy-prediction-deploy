"""
Deep Neural Network (MLP) for High-Risk Pregnancy Prediction

Feedforward Multi-Layer Perceptron using TensorFlow/Keras Dense layers.
This is the appropriate deep learning architecture for tabular (non-sequential)
clinical data. An LSTM would require temporal/sequential structure which this
dataset does not possess.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


def prepare_input(X):
    """Convert DataFrame to numpy array for Keras input."""
    if isinstance(X, pd.DataFrame):
        return X.values.astype(np.float32)
    return X.astype(np.float32)


def build_model(n_features, learning_rate=0.001):
    """
    Build an MLP model for binary classification.

    Parameters
    ----------
    n_features : int
        Number of input features.
    learning_rate : float
        Learning rate for the Adam optimizer.

    Returns
    -------
    tf.keras.Model
    """
    model = Sequential([
        Dense(64, activation="relu", input_shape=(n_features,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(16, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def get_callbacks():
    """Return standard training callbacks."""
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
    )

    return [early_stopping, reduce_lr]


def train_model(model, X_train, y_train, X_val, y_val,
                epochs=100, batch_size=32, class_weight=None):
    """
    Train the MLP model.

    Parameters
    ----------
    model : tf.keras.Model
    X_train, y_train : Training data (2D numpy arrays).
    X_val, y_val : Validation data (2D numpy arrays).
    epochs : int
    batch_size : int
    class_weight : dict or None

    Returns
    -------
    history : tf.keras.callbacks.History
    """
    callbacks = get_callbacks()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    return history


def predict(model, X, threshold=0.5):
    """Generate binary predictions from the MLP model."""
    probabilities = model.predict(X).flatten()
    predictions = (probabilities >= threshold).astype(int)
    return predictions, probabilities


def save_model(model, path):
    """Save trained Keras model."""
    model.save(path)
    print(f"MLP model saved to {path}")


def load_model(path):
    """Load trained Keras model."""
    return tf.keras.models.load_model(path)
