import numpy as np
from helpers_perso import *
from helpers_perso.helpers_nan_imputation import identify_integer_columns


def identify_categorical_columns(X, treshold):
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

    unique_value_counts = np.array(
        [len(np.unique(X[:, col])) for col in integer_columns]
    )
    # print(unique_value_counts)

    indexes_categorical_features = [
        i for i, count in enumerate(unique_value_counts) if count <= treshold
    ]

    indexes_non_categorical_features = [
        i
        for i in range(len(unique_value_counts))
        if i not in indexes_categorical_features
    ]

    assert len(indexes_categorical_features) + len(
        indexes_non_categorical_features
    ) == len(unique_value_counts)
    assert unique_value_counts.size == len(integer_columns)

    return indexes_categorical_features, indexes_non_categorical_features


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
