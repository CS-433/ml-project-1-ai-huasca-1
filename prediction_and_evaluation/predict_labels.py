import numpy as np
from helpers_perso.helpers_implementations import sigmoid


def predict_classification(x_test, w, treshold=0):
    """
    Predict binary classification labels using the weights from a linear regression model.

    Args:
        x_test (np.ndarray): The input data of shape (n_samples, n_features).
        w (np.ndarray): The weight vector of shape (n_features,).

    Returns:
        np.ndarray: Predicted binary labels (1 or -1) of shape (n_samples,).
    """
    # Compute the linear combination of inputs and weights
    y_pred = np.dot(x_test, w)

    # Apply the threshold to obtain binary classification
    y_test = np.where(y_pred >= treshold, 1, -1)

    return y_test


def predict_classification_logistic(x_test, w, treshold=0.5):
    """
    Predict binary classification labels using the weights from a linear regression model.

    Args:
        x_test (np.ndarray): The input data of shape (n_samples, n_features).
        w (np.ndarray): The weight vector of shape (n_features,).

    Returns:
        np.ndarray: Predicted binary labels (1 or -1) of shape (n_samples,).
    """
    # Compute the linear combination of inputs and weights
    y_pred = sigmoid(np.dot(x_test, w))

    # Apply the threshold to obtain binary classification
    y_test = np.where(y_pred >= treshold, 1, -1)

    return y_test
