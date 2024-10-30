import numpy as np


def one_hot_encode_columns(arr, column_indices):
    """
    Perform one-hot encoding on specific columns of a 2D NumPy array, treating NaN as a separate category using only NumPy.

    Args:
        arr (np.ndarray): A 2D NumPy array.
        column_indices (list of int): A list of column indices to one-hot encode.

    Returns:
        np.ndarray: A 2D NumPy array with the one-hot encoded columns and the rest of the data.
    """
    # Sort column indices in descending order to avoid shifting issues when removing columns
    column_indices = sorted(column_indices, reverse=True)

    # Initialize the result as a copy of the input array to avoid modifying the original data
    result = np.delete(
        arr, column_indices, axis=1
    )  # Start with the array without the specified columns

    count = 0
    # Loop over each column index to one-hot encode
    for column_index in column_indices:
        count += 1
        # Extract the column to be one-hot encoded
        column_to_encode = arr[:, column_index]

        # Handle NaN values by treating them as a separate category
        column_to_encode = np.where(np.isnan(column_to_encode), "NaN", column_to_encode)

        # Get the unique values (including 'NaN') and create a mapping
        unique_values = np.unique(column_to_encode)
        value_to_index = {value: idx for idx, value in enumerate(unique_values)}

        # Create a zero matrix for the one-hot encoding with shape (len(arr), number of unique values)
        one_hot_encoded = np.zeros((arr.shape[0], unique_values.shape[0]), dtype=int)

        # Populate the one-hot matrix
        for i, value in enumerate(column_to_encode):
            one_hot_encoded[i, value_to_index[value]] = 1

        # Concatenate the one-hot encoded columns to the result array
        one_hot_encoded_arr = np.concatenate([result, one_hot_encoded], axis=1)

        print(
            f"Column {column_index+1} ({count}/{len(column_indices)}) one-hot encoded successfully!"
        )

    return one_hot_encoded_arr


def binary_encode_columns(arr, column_indices):
    """
    Perform binary encoding on specific columns of a 2D NumPy array.

    Args:
        arr (np.ndarray): A 2D NumPy array.
        column_indices (list of int): A list of column indices to binary encode.

    Returns:
        np.ndarray: A 2D NumPy array with the binary encoded columns and the rest of the data.
    """
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
