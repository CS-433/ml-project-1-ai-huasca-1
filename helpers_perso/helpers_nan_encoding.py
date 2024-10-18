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

def remove_nan_features(array, min_proportion=0.8):
    """
    Removes columns containing NaN values from a given array if the proportion of NaNs is greater than min_proportion.
    Prints the percentage of columns deleted.

    Args:
        array (numpy.ndarray): The input array to clean.
        min_proportion (float): The minimum proportion of NaNs required to remove a column.

    Returns:
        numpy.ndarray: The cleaned array with NaN-containing columns removed.
    """
    # Calculate the proportion of NaNs in each column
    nan_proportions = np.isnan(array).sum(axis=0) / array.shape[0]

    # Identify columns that contain NaN proportions greater than min_proportion
    cols_to_remove = nan_proportions > min_proportion

    # Calculate the percentage of columns to be removed
    percentage_deleted = np.sum(cols_to_remove) / array.shape[1] * 100
    print(f"Percentage of columns to delete (NaN proportion superior to {min_proportion*100} %): {percentage_deleted:.2f}%")

    # Remove columns containing NaN proportions greater than min_proportion
    cleaned_array = array[:, ~cols_to_remove]
    print("Data cleaned successfully!")
    print(f"Original shape of x_train: {array.shape}")
    print(f"Cleaned shape of x_train: {cleaned_array.shape}")

    return cleaned_array

def encode_nan_integer_columns(arr, replacement_value='zero'):
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

    count = 0
    # Iterate over each column
    for col_index in range(arr.shape[1]):
        column = arr[:, col_index]
        
        # Check if the column contains only integers (ignoring NaN values)
        is_integer_column = np.all(np.isnan(column) | np.equal(np.mod(column, 1), 0))
        
        # If the column meets the criteria (integer-only, no zeroes)
        if is_integer_column:
            count+=1
            
            # If NaN proportion is too high, mark the column for deletion
            if replacement_value == 'zero':
                # Replace NaN values with 0
                arr[:, col_index] = np.where(np.isnan(column), 0, column)
                print(f"Column {col_index} has been encoded with NaNs as 0")
            elif replacement_value == 'upper':
                # Get the unique values in the column, excluding NaN
                unique_values = np.unique(column[~np.isnan(column)])
                # Replace NaN with N+1, where N is the number of unique values
                arr[:, col_index] = np.where(np.isnan(column), len(unique_values) + 1, column)
                print(f"Column {col_index} has been encoded with NaNs as {unique_values.shape[0] + 1}")
            elif replacement_value == 'mode':
                # Replace NaN with the mode of the column
                mode = calculate_mode_integer(column)
                arr[:, col_index] = np.where(np.isnan(column), mode, column)
                print(f"Column {col_index} has been encoded with NaNs as the mode {mode}")

    print(f"Number of columns encoded: {count}")
    
    return arr

def encode_nan_continuous_columns(arr, replacement_value='zero'):
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

    count = 0
    # Iterate over each column
    for col_index in range(arr.shape[1]):
        column = arr[:, col_index]
        
        # Check if the column contains only integers (ignoring NaN values)
        is_not_integer_column= np.any(~np.isnan(column) & (np.mod(column, 1) != 0))
        
        # If the column meets the criteria (integer-only, no zeroes)
        if is_not_integer_column:
            count+=1
            
            # If NaN proportion is too high, mark the column for deletion
            if replacement_value == 'zero':
                # Replace NaN values with 0
                arr[:, col_index] = np.where(np.isnan(column), 0, column)
                print(f"Column {col_index} has been encoded with NaNs as 0")
            elif replacement_value == 'mean':
                # Calculate the mean of the column, excluding NaN
                mean_value = np.nanmean(column)
                # Replace NaN with the mean of the column
                arr[:, col_index] = np.where(np.isnan(column), mean_value, column)
                print(f"Column {col_index} has been encoded with NaNs as the mean {mean_value}")
            elif replacement_value == 'mode':
                # Replace NaN with the binned mode of the column
                mode = calculate_mode_binned(column)
                arr[:, col_index] = np.where(np.isnan(column), mode, column)
                print(f"Column {col_index} has been encoded with NaNs as the binned mode {mode}")

    print(f"Number of columns encoded: {count}")
    
    return arr