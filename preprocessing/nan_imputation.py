import numpy as np
from helpers_perso.helpers_nan_imputation import (
    calculate_mode_integer,
    calculate_mode_binned,
)


def remove_nan_features(X, min_proportion):
    """
    Removes columns containing NaN values from a given array if the proportion of NaNs is greater than min_proportion.
    Prints the percentage of columns deleted and returns the cleaned array along with the indices of deleted columns.

    Args:
        X (numpy.ndarray): The input array to clean.
        min_proportion (float): The minimum proportion of NaNs required to remove a column.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: The cleaned array with NaN-containing columns removed.
            - list of int: The indices of the columns that were deleted.
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
    Encode NaN values in columns that contain only integers and do not contain zeroes.
    If `as_zero` is True, replace NaN values with 0. Otherwise, replace NaN values with N+1,
    where N is the number of unique values in the column.

    Delete the columns containing a proportion of NaN values greater than `max_proportion`.

    Args:
        arr (np.ndarray): A 2D NumPy array.
        max_proportion (float): Maximum proportion of NaN values allowed in a column before it is deleted.
        as_zero (bool): If True, encode NaNs as zero. If False, encode NaNs as N+1, where N is the number of unique values.

    Returns:
        np.ndarray: A 2D NumPy array with NaN values replaced by zeroes or N+1 where applicable.
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
    Encode NaN values in columns that contain only integers and do not contain zeroes.
    If `as_zero` is True, replace NaN values with 0. Otherwise, replace NaN values with N+1,
    where N is the number of unique values in the column.

    Delete the columns containing a proportion of NaN values greater than `max_proportion`.

    Args:
        arr (np.ndarray): A 2D NumPy array.
        max_proportion (float): Maximum proportion of NaN values allowed in a column before it is deleted.
        as_zero (bool): If True, encode NaNs as zero. If False, encode NaNs as N+1, where N is the number of unique values.

    Returns:
        np.ndarray: A 2D NumPy array with NaN values replaced by zeroes or N+1 where applicable.
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
                mode = calculate_mode_binned(column,200)
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
