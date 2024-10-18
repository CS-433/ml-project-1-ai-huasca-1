import numpy as np


def one_hot_encode_column(arr, column_index):
    """
    Perform one-hot encoding on a specific column of a 2D NumPy array, treating NaN as a separate category using only NumPy.

    Args:
        arr (np.ndarray): A 2D NumPy array.
        column_index (int): The index of the column to one-hot encode.

    Returns:
        np.ndarray: A 2D NumPy array with the one-hot encoded column and the rest of the data.
    """
    # Extract the column to be one-hot encoded
    column_to_encode = arr[:, column_index]

    # Handle NaN values by treating them as a separate category
    # Convert NaNs to a string or placeholder value ('NaN') that can be indexed
    column_to_encode = np.where(np.isnan(column_to_encode), "NaN", column_to_encode)

    # Get the unique values (including 'NaN') and their corresponding indices
    unique_values = np.unique(column_to_encode)

    # Create a mapping from each unique value to an index
    value_to_index = {value: idx for idx, value in enumerate(unique_values)}

    # Initialize a zero matrix with shape (len(arr), number of unique values)
    one_hot_encoded = np.zeros((arr.shape[0], unique_values.shape[0]), dtype=int)

    # Populate the one-hot matrix
    for i, value in enumerate(column_to_encode):
        one_hot_encoded[i, value_to_index[value]] = 1

    # Remove the original column and concatenate the one-hot encoded columns
    arr_without_column = np.delete(arr, column_index, axis=1)
    one_hot_encoded_arr = np.concatenate([arr_without_column, one_hot_encoded], axis=1)

    return one_hot_encoded_arr
