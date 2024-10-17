import numpy as np

# PREPROCESSING


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

def encode_nan_as_zero_integer_columns(arr, max_proportion):
    """
    Encode NaN values as zeroes in columns that contain only integers and do not contain zeroes.
    Delete the columns containing a proportion of Nan values greater than `max_proportion`, 
    or with a number of unique values outside the range [`min_unique`, `max_unique`].
    
    Args:
        arr (np.ndarray): A 2D NumPy array.
        min_unique (int): Minimum number of unique values a column must have to encode NaNs as zeroes.
        max_unique (int): Maximum number of unique values a column must have to encode NaNs as zeroes.
        
    Returns:
        np.ndarray: A 2D NumPy array with NaN values replaced by zeroes where applicable.
    """
    # Iterate over each column
    for col_index in range(arr.shape[1]):
        column = arr[:, col_index]
        
        # Check if the column contains only integers (ignoring NaN values)
        is_integer_column = np.all(np.isnan(column) | np.equal(np.mod(column, 1), 0))
        
        # Check if the column contains any zero
        contains_zero = np.any(column == 0)
        
        # If the column meets the criteria, replace NaNs with 0
        if is_integer_column and not contains_zero:

            nan_proportion = np.isnan(column).sum() / len(column)
            
            if nan_proportion > max_proportion:
                arr = np.delete(arr, col_index, axis=1)
            else:    
                # Replace NaN values with 0
                arr[:, col_index] = np.where(np.isnan(column), 0, column)
    
    return arr

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
    column_to_encode = np.where(np.isnan(column_to_encode), 'NaN', column_to_encode)
    
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

# LINEAR REGRESSION WITH GRADIENT DESCENT

def compute_loss(y, tx, w):
    """Calculate the mean squared error (MSE) loss.

    Args:
        y (numpy.ndarray): Target values of shape (N,).
        tx (numpy.ndarray): Input data of shape (N, D).
        w (numpy.ndarray): Model parameters of shape (D,).

    Returns:
        float: The value of the MSE loss.
    """
    # Compute the error vector (difference between actual and predicted values)
    error = y - tx.dot(w)

    # Compute the mean squared error (MSE) loss
    mse = np.mean(error**2) / 2

    return mse

def compute_gradient(y, tx, w):
    """Compute the gradient of the loss.

    Args:
        y (numpy.ndarray): Target values of shape (N,).
        tx (numpy.ndarray): Input data of shape (N, D).
        w (numpy.ndarray): Model parameters of shape (D,).

    Returns:
        numpy.ndarray: Gradient of the loss with respect to w, of shape (D,).
    """
    # Compute the error vector (difference between actual and predicted values)
    error = y - tx.dot(w)

    # Compute the gradient of the loss function
    gradient = -tx.T.dot(error) / y.size

    return gradient

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Perform gradient descent optimization.

    Args:
        y (numpy.ndarray): Target values of shape (N,).
        tx (numpy.ndarray): Input data of shape (N, D).
        initial_w (numpy.ndarray): Initial guess for the model parameters of shape (D,).
        max_iters (int): Total number of iterations for gradient descent.
        gamma (float): Step size (learning rate) for gradient updates.

    Returns:
        numpy.ndarray: Final weight vector of shape (D,).
        float: Final loss (MSE) value.
    """
    # Initialize lists to store weights and losses at each iteration
    ws = [initial_w]
    initial_loss = compute_loss(y, tx, initial_w)
    losses = [initial_loss]
    w = initial_w

    # Iterate over the number of iterations
    for n_iter in range(max_iters):
        # Compute the gradient of the loss function
        gradient = compute_gradient(y, tx, w)
        # Update the weights using the gradient and learning rate
        w -= gamma * gradient
        # Compute the loss with the updated weights
        loss = compute_loss(y, tx, w)
        # Store the updated weights and loss
        ws.append(w)
        losses.append(loss)
        # Print the current iteration, loss, and weights
        print(f"GD iter. {n_iter}/{max_iters - 1}: loss={loss}, w0={w[0]}, w1={w[1]}")

    # Compute the final loss
    loss = compute_loss(y, tx, w)
    return w, loss

# LINEAR REGRESSION WITH STOCHASTIC GRADIENT DESCENT

def batch_iter(y, tx, batch_size, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.

    Args:
        y (numpy.ndarray): The output desired values of shape (N,).
        tx (numpy.ndarray): The input data of shape (N, D).
        batch_size (int): The size of each mini-batch.
        shuffle (bool): Whether to shuffle the data before creating mini-batches.

    Yields:
        tuple: Mini-batches of (y, tx) of shape (batch_size,) and (batch_size, D) respectively.
    """
    data_size = len(y)  # Number of data points.

    if shuffle:
        idxs = np.random.permutation(data_size)  # Shuffle indices
    else:
        idxs = np.arange(data_size)  # Create ordered indices

    for start in range(0, data_size, batch_size):
        end = min(
            start + batch_size, data_size
        )  # Determine the end index of the current batch
        yield y[idxs[start:end]], tx[idxs[start:end]]  # Yield the mini-batch


def compute_stoch_gradient(y, tx, w):
    """Compute the stochastic gradient of the loss at w for a mini-batch of data.

    Args:
        y: numpy array of shape (B,), the target values for the mini-batch.
        tx: numpy array of shape (B, D), the input data for the mini-batch.
        w: numpy array of shape (D,), the current weight vector.

    Returns:
        numpy array of shape (D,), the stochastic gradient of the loss at w.
    """
    # Compute predictions for the mini-batch
    predictions = tx.dot(w)

    # Compute the error vector (difference between actual and predicted values)
    e = y - predictions

    # Compute the stochastic gradient of the loss function
    stoch_gradient = -(1 / y.size) * np.dot(tx.T, e)

    return stoch_gradient


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Perform stochastic gradient descent (SGD) optimization.

    Args:
        y (numpy.ndarray): Target values of shape (N,).
        tx (numpy.ndarray): Input data of shape (N, D).
        initial_w (numpy.ndarray): Initial guess for the model parameters of shape (D,).
        batch_size (int): Number of data points in a mini-batch used for computing the stochastic gradient.
        max_iters (int): Total number of iterations for SGD.
        gamma (float): Step size (learning rate) for gradient updates.

    Returns:
        numpy.ndarray: Final weights of shape (D,).
        float: Final loss value.
    """
    # Initialize the weights with the initial guess
    w = initial_w

    # Iterate over the number of iterations
    for n_iter in range(max_iters):
        # Generate mini-batches and perform updates
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            # Compute the stochastic gradient for the current mini-batch
            gradient = compute_stoch_gradient(minibatch_y, minibatch_tx, w)
            # Update the weights using the gradient and learning rate
            w -= gamma * gradient

    # Compute the final loss using the updated weights
    loss = compute_loss(y, tx, w)

    # Return the final weights and loss
    return w, loss

# LOGISTIC REGRESSION

def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array

    >>> sigmoid(np.array([0.1]))
    array([0.52497919])
    >>> sigmoid(np.array([0.1, 0.1]))
    array([0.52497919, 0.52497919])
    """

    sigmoid_function = 1 / (1 + np.exp(-t))

    return sigmoid_function


def calculate_loss_sigmoid(y, tx, w):
    """Compute the cost by negative log likelihood.

    Args:
        y: numpy array of shape (N,), the target values (0 or 1).
        tx: numpy array of shape (N, D), the input data.
        w: numpy array of shape (D,), the weight vector.

    Returns:
        loss: scalar, the negative log likelihood loss.

    >>> y = np.array([0., 1.])
    >>> tx = np.arange(4).reshape(2, 2)
    >>> w = np.array([2., 3.])
    >>> round(calculate_loss_sigmoid(y, tx, w), 8)
    1.52429481
    """
    # Ensure the number of samples matches between y and tx
    assert y.shape[0] == tx.shape[0]
    # Ensure the number of features matches between tx and w
    assert tx.shape[1] == w.shape[0]

    # Compute the negative log likelihood loss
    loss = -np.mean(
        y * np.log(sigmoid(np.dot(tx, w)))  # Contribution of positive class
        + (1 - y) * np.log(1 - sigmoid(np.dot(tx, w)))  # Contribution of negative class
    )
    return loss


def calculate_gradient_sigmoid(y, tx, w):
    """Compute the gradient of the loss for logistic regression.

    Args:
        y: numpy array of shape (N,), the target values (0 or 1).
        tx: numpy array of shape (N, D), the input data.
        w: numpy array of shape (D,), the current weight vector.

    Returns:
        gradient: numpy array of shape (D,), the gradient of the loss.

    >>> np.set_printoptions(8)
    >>> y = np.array([0., 1.])
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([0.1, 0.2, 0.3])
    >>> calculate_gradient_sigmoid(y, tx, w)
    array([-0.10370763,  0.2067104 ,  0.51712843])
    """
    # Compute the dot product of tx and w, then apply the sigmoid function
    sigmoid_tx_w = sigmoid(np.dot(tx, w))

    # Compute the difference between the sigmoid predictions and the actual target values
    error = sigmoid_tx_w - y

    # Compute the gradient of the loss function
    gradient = np.dot(tx.T, error) / y.shape[0]

    return gradient


def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Perform one step of gradient descent using logistic regression. Return the updated weights and the loss.

    Args:
        y: numpy array of shape (N,), the target values (0 or 1).
        tx: numpy array of shape (N, D), the input data.
        w: numpy array of shape (D,), the current weight vector.
        gamma: float, the step size (learning rate) for weight updates.

    Returns:
        w: numpy array of shape (D,), the updated weight vector.
        loss: scalar, the loss value after the update.

    >>> y = np.array([0., 1.])
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([0.1, 0.2, 0.3])
    >>> gamma = 0.1
    >>> w, loss = learning_by_gradient_descent(y, tx, w, gamma)
    >>> round(loss, 8)
    0.62137268
    >>> w
    array([0.11037076, 0.17932896, 0.24828716])
    """
    # Compute the gradient of the loss function
    gradient = calculate_gradient_sigmoid(y, tx, w)

    # Update the weights using the gradient and learning rate
    w = w - gamma * gradient

    # Compute the loss with the updated weights
    loss = calculate_loss_sigmoid(y, tx, w)

    return w, loss


