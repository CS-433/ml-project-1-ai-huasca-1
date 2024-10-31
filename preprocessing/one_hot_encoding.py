import numpy as np
from helpers_perso import *
from helpers_perso.helpers_nan_imputation import identify_integer_columns


def binary_encode_columns(X, column_indices):
    """
    Perform binary encoding on specific columns of a 2D NumPy array.

    Args:
        arr (np.ndarray): A 2D NumPy array.
        column_indices (list of int): A list of column indices to binary encode.

    Returns:
        np.ndarray: A 2D NumPy array with the binary encoded columns and the rest of the data.
    """
    arr = np.copy(X)
    # Sort column indices in descending order to avoid shifting issues when removing columns
    column_indices = sorted(column_indices, reverse=True)

    # Collect the non-categorical columns (those not in column_indices)
    non_categorical_columns = np.delete(arr, column_indices, axis=1)

    # Initialize a list to store binary encoded columns
    encoded_columns = [non_categorical_columns]

    # Process each column index
    for column_index in column_indices:
        # Extract the column to be binary encoded
        column_to_encode = arr[:, column_index]

        # Handle NaN values by treating them as a separate category
        column_to_encode = np.where(np.isnan(column_to_encode), "NaN", column_to_encode)

        # Get unique values and assign a binary code to each unique category
        unique_values, encoded_indices = np.unique(
            column_to_encode, return_inverse=True
        )

        # Determine the number of bits needed to represent each unique value
        num_bits = int(np.ceil(np.log2(len(unique_values))))

        # Convert each index to binary and split into separate columns
        binary_matrix = (
            (encoded_indices[:, None] & (1 << np.arange(num_bits))) > 0
        ).astype(int)

        # Append the binary encoded matrix for this column
        encoded_columns.append(binary_matrix)

    # Concatenate all encoded columns along with non-categorical data
    binary_encoded_arr = np.concatenate(encoded_columns, axis=1)

    return binary_encoded_arr


def identify_categorical_columns(X,treshold):
    """
    Identify categorical columns in a dataset based on a threshold for unique values.
    Parameters:
    X (numpy.ndarray): The input data array where rows are samples and columns are features.
    treshold (int): The maximum number of unique values a column can have to be considered categorical.
    Returns:
    tuple: A tuple containing two lists:
        - indexes_categorical_features (list): Indices of columns identified as categorical.
        - indexes_non_categorical_features (list): Indices of columns identified as non-categorical.
    """

    integer_columns, non_integer_columns = identify_integer_columns(X)

    # print(integer_columns)

    unique_value_counts = np.array([len(np.unique(X[:, col])) for col in integer_columns])
    # print(unique_value_counts)

    indexes_categorical_features = [i for i, count in enumerate(unique_value_counts) if count <= treshold]

    indexes_non_categorical_features = [i for i in range(len(unique_value_counts)) if i not in indexes_categorical_features]

    assert len(indexes_categorical_features) + len(indexes_non_categorical_features) == len(unique_value_counts)
    assert unique_value_counts.size == len(integer_columns)

    return indexes_categorical_features,indexes_non_categorical_features

import numpy as np



def binary_encode_column(column, unique_values):
    """
    Binary encodes a column based on a given set of unique values.

    Args:
        column (np.ndarray): The column to be encoded.
        unique_values (list): The list of unique values to use for encoding.

    Returns:
        np.ndarray: Binary encoded column with consistent structure.
    """
    # Map each unique value to an index
    value_to_index = {value: idx for idx, value in enumerate(unique_values)}

    # Create a binary encoded matrix for this column
    num_bits = int(np.ceil(np.log2(len(unique_values))))
    binary_encoded = np.zeros((len(column), num_bits), dtype=int)
    
    # Encode each value in the column
    for i, value in enumerate(column):
        if value in value_to_index:
            index = value_to_index[value]
        else:
            index = len(unique_values)  # Handle unseen values with an extra encoding
        binary_encoded[i] = np.array(list(np.binary_repr(index, num_bits)), dtype=int)

    return binary_encoded



def consistent_binary_encode(X_train, X_test, categorical_columns):
    """
    Binary encode categorical columns in X_train and apply the same encoding to X_test.

    Args:
        X_train (np.ndarray): Training data array.
        X_test (np.ndarray): Test data array.
        categorical_columns (list): List of column indices for categorical features.

    Returns:
        tuple: Binary encoded versions of X_train and X_test with consistent structure.
    """
    X_train_encoded = []
    X_test_encoded = []

    for col in range(X_train.shape[1]):
        if col in categorical_columns:
            # Get unique values from X_train for this column
            unique_values = np.unique(X_train[:, col][~np.isnan(X_train[:, col])])
            
            # Binary encode the column for both X_train and X_test
            X_train_encoded.append(binary_encode_column(X_train[:, col], unique_values))
            X_test_encoded.append(binary_encode_column(X_test[:, col], unique_values))
        else:
            # Keep non-categorical columns as-is
            X_train_encoded.append(X_train[:, col].reshape(-1, 1))
            X_test_encoded.append(X_test[:, col].reshape(-1, 1))

    # Concatenate all columns to get the final encoded arrays
    X_train_encoded = np.hstack(X_train_encoded)
    X_test_encoded = np.hstack(X_test_encoded)

    return X_train_encoded, X_test_encoded
