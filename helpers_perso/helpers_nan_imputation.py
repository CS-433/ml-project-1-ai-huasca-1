import numpy as np

def calculate_mode_integer(column):
    """
    Calculate the mode of a column containing only integer values.

    Args:
        column (np.ndarray): A 1D NumPy array containing integer values.

    Returns:
        int: The most frequent value in the column, representing the mode.
    """
    # Remove NaN values for mode calculation
    column = column[~np.isnan(column)]

    if len(column) == 0:
        raise ValueError("The column contains only NaN values, cannot compute mode.")

    # Get the unique values and their corresponding counts
    unique_values, counts = np.unique(column, return_counts=True)

    # Find the index of the value that has the highest count
    max_count_index = np.argmax(counts)

    # Return the unique value with the highest count (the mode)
    return unique_values[max_count_index]


def calculate_mode_binned(column, num_bins=100):
    """
    Calculate the mode of a column containing rational values by dividing its range into 100 bins.

    Args:
        column (np.ndarray): A 1D NumPy array containing rational values.
        num_bins (int): The number of bins to divide the range into. Default is 100.

    Returns:
        float: The midpoint of the bin with the most values, representing the mode.
    """
    # Remove NaN values for mode calculation
    column = column[~np.isnan(column)]

    if len(column) == 0:
        raise ValueError("The column contains only NaN values, cannot compute mode.")

    # Find the minimum and maximum values of the column
    col_min = np.min(column)
    col_max = np.max(column)

    # Create bins (num_bins + 1 to include both edges of the range)
    bins = np.linspace(col_min, col_max, num_bins + 1)

    # Digitize the column values into bins (returns bin indices)
    bin_indices = np.digitize(column, bins)

    # Count occurrences in each bin
    bin_counts = np.bincount(bin_indices, minlength=num_bins + 1)

    # Find the bin with the highest count
    max_bin_index = np.argmax(bin_counts)

    # Calculate the midpoint of the bin with the most values
    mode_bin_midpoint = (bins[max_bin_index - 1] + bins[max_bin_index]) / 2

    return mode_bin_midpoint

def identify_integer_columns(X):
    """
    Identify columns that contain only integers and those that contain non-integer values.

    Args:
        X (np.ndarray): The input array of shape (n_samples, n_features).

    Returns:
        tuple: A tuple containing:
            - integer_columns (list): A list of column indices that contain only integers.
            - non_integer_columns (list): A list of column indices that contain non-integer values.
    """
    integer_columns = []
    non_integer_columns = []

    for i in range(X.shape[1]):
        # Remove NaNs and check if all values are integers
        col_without_nan = X[:, i][~np.isnan(X[:, i])]
        
        if np.all(np.mod(col_without_nan, 1) == 0):
            integer_columns.append(i)
        else:
            non_integer_columns.append(i)

    assert len(integer_columns) + len(non_integer_columns) == X.shape[1]
    assert len(set(integer_columns).intersection(non_integer_columns)) == 0, "An element is found in both lists"

    return integer_columns, non_integer_columns