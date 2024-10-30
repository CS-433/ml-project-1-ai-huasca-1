import numpy as np


def standardize_columns(arr, indices):
    """
    Standardize specified columns in a 2D array, without modifying other columns.

    Args:
        arr (np.ndarray): The input array of shape (n_samples, n_features).
        indices (list of int): List of column indices to standardize.

    Returns:
        np.ndarray: The array with specified columns standardized.
    """
    arr = arr.copy()  # Create a copy to avoid modifying the original array

    for index in indices:
        column = arr[:, index]
        
        # Compute mean and standard deviation
        mean = np.mean(column)
        std = np.std(column)

        # Check for zero standard deviation to avoid division by zero
        if std == 0:
            std = 1
        
        # Standardize the column
        arr[:, index] = (column - mean) / std

    return arr
