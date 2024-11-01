import numpy as np
from helpers_perso.helpers_nan_imputation import (
    calculate_mode_integer,
    calculate_mode_binned,
)


def remove_nan_features(X, min_proportion):
    """
    Removes columns from the array `X` where the proportion of NaN values exceeds `min_proportion`.
    This function outputs the percentage of columns removed, along with details on the original
    and cleaned array shapes, and returns the cleaned array and indices of the deleted columns.

    Args:
        X (numpy.ndarray): The input 2D array to clean by removing columns with high NaN proportions.
        min_proportion (float): The threshold proportion of NaN values required to remove a column.
                                Columns with a NaN proportion exceeding this value will be removed.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: The cleaned array with columns containing excessive NaNs removed.
            - list of int: The indices of the columns that were deleted.

    Notes:
        - Prints the percentage of columns deleted based on the `min_proportion` threshold.
        - Displays the original and new shapes of the array after cleaning.
    """
    array = np.copy(X)
    # Calculate the proportion of NaNs in each column
    nan_proportions = np.isnan(array).sum(axis=0) / array.shape[0]

    # Identify columns that contain NaN proportions greater than min_proportion
    cols_to_remove = nan_proportions > min_proportion
    deleted_indices = np.where(cols_to_remove)[0]

    # Calculate the percentage of columns to be removed
    percentage_deleted = np.sum(cols_to_remove) / array.shape[1] * 100
    print(
        f"Percentage of columns to delete (NaN proportion superior to {min_proportion*100} %): {percentage_deleted:.2f}%"
    )

    # Remove columns containing NaN proportions greater than min_proportion
    cleaned_array = array[:, ~cols_to_remove]
    print("Data cleaned successfully!")
    print(f"Original shape of x_train: {array.shape}")
    print(f"Cleaned shape of x_train: {cleaned_array.shape}")

    return cleaned_array, deleted_indices


def encode_nan_integer_columns(X, replacement_value="zero"):
    """
    Encodes NaN values in columns containing only integers (excluding zeroes). Depending on
    the specified replacement strategy, NaN values are replaced either by zero, by the count
    of unique values plus one (N+1), or by the column mode.

    Args:
        X (np.ndarray): A 2D NumPy array where columns containing NaN values may be encoded.
        replacement_value (str): Strategy for replacing NaN values:
            - "zero": Replace NaN values with 0.
            - "upper": Replace NaN values with N+1, where N is the count of unique values
                       in the column (excluding NaN).
            - "mode": Replace NaN values with the most frequent (mode) value in the column.

    Returns:
        np.ndarray: A modified 2D NumPy array where NaN values in integer-only columns have been
                    replaced according to the specified `replacement_value`.

    Notes:
        - Only columns containing integer values (without zeroes) are processed.
        - Columns without NaN values are not affected.
        - Prints the number of columns processed and the encoding applied per column.
    """
    arr = np.copy(X)
    count = 0
    # Iterate over each column
    for col_index in range(arr.shape[1]):
        column = arr[:, col_index]

        # Check if the column contains any NaN values
        if not np.any(np.isnan(column)):
            continue

        # Check if the column contains only integers (ignoring NaN values)
        is_integer_column = np.all(np.isnan(column) | np.equal(np.mod(column, 1), 0))

        # If the column meets the criteria (integer-only, no zeroes)
        if is_integer_column:
            count += 1

            if replacement_value == "zero":
                # Replace NaN values with 0
                arr[:, col_index] = np.where(np.isnan(column), 0, column)
                print(f"Column {col_index} has been encoded with NaNs as 0")
            elif replacement_value == "upper":
                # Get the unique values in the column, excluding NaN
                unique_values = np.unique(column[~np.isnan(column)])
                # Replace NaN with N+1, where N is the number of unique values
                arr[:, col_index] = np.where(
                    np.isnan(column), len(unique_values) + 1, column
                )
                print(
                    f"Column {col_index} has been encoded with NaNs as {unique_values.shape[0] + 1}"
                )
            elif replacement_value == "mode":
                # Replace NaN with the mode of the column
                mode = calculate_mode_integer(column)
                arr[:, col_index] = np.where(np.isnan(column), mode, column)
                print(
                    f"Column {col_index} has been encoded with NaNs as the mode {mode}"
                )

    print(f"Number of integer columns encoded: {count}")

    return arr


def encode_nan_continuous_columns(X, replacement_value="zero"):
    """
    Encodes NaN values in columns containing continuous data (non-integer values).
    Based on the specified replacement strategy, NaN values are replaced with zero,
    the column mean, the binned mode, or the column median.

    Args:
        X (np.ndarray): A 2D NumPy array containing continuous data columns where NaN
                        values may be encoded.
        replacement_value (str): Strategy for replacing NaN values:
            - "zero": Replace NaN values with 0.
            - "mean": Replace NaN values with the column mean (ignoring NaNs).
            - "mode": Replace NaN values with the most frequent (binned mode) value.
            - "median": Replace NaN values with the column median (ignoring NaNs).

    Returns:
        np.ndarray: A modified 2D NumPy array where NaN values in continuous columns have
                    been replaced according to the specified `replacement_value`.

    Notes:
        - Only columns containing continuous (non-integer) values are processed.
        - Columns without NaN values are not affected.
        - Prints the number of columns processed and the encoding method applied per column.
    """
    arr = np.copy(X)

    count = 0
    # Iterate over each column
    for col_index in range(arr.shape[1]):
        column = arr[:, col_index]

        # Check if the column contains any NaN values
        if not np.any(np.isnan(column)):
            continue

        # Check if the column contains only integers (ignoring NaN values)
        is_not_integer_column = np.any(~np.isnan(column) & (np.mod(column, 1) != 0))

        # If the column meets the criteria (integer-only, no zeroes)
        if is_not_integer_column:
            count += 1

            # If NaN proportion is too high, mark the column for deletion
            if replacement_value == "zero":
                # Replace NaN values with 0
                arr[:, col_index] = np.where(np.isnan(column), 0, column)
                print(f"Column {col_index} has been encoded with NaNs as 0")
            elif replacement_value == "mean":
                # Calculate the mean of the column, excluding NaN
                mean_value = np.nanmean(column)
                # Replace NaN with the mean of the column
                arr[:, col_index] = np.where(np.isnan(column), mean_value, column)
                print(
                    f"Column {col_index} has been encoded with NaNs as the mean {mean_value}"
                )
            elif replacement_value == "mode":
                # Replace NaN with the binned mode of the column
                mode = calculate_mode_binned(column, 200)
                arr[:, col_index] = np.where(np.isnan(column), mode, column)
                print(
                    f"Column {col_index} has been encoded with NaNs as the binned mode {mode}"
                )
            elif replacement_value == "median":
                # Calculate the median of the column, excluding NaN
                median_value = np.nanmedian(column)
                # Replace NaN with the median of the column
                arr[:, col_index] = np.where(np.isnan(column), median_value, column)
                print(
                    f"Column {col_index} has been encoded with NaNs as the median {median_value}"
                )

    print(f"Number of non integer columns encoded: {count}")

    return arr
