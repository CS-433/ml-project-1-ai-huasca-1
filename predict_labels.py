import numpy as np

def predict_classification(x_test, w):
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
    y_test = np.where(y_pred >= 0, 1, -1)
    
    return y_test