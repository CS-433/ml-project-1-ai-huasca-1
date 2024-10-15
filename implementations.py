import os
from helpers import load_csv_data
import numpy as np
import matplotlib.pyplot as plt


### IMPLEMENTATION 1


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


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent (GD) with MSE as the loss function.

    Args:
        y (numpy.ndarray): Target values of shape (N,).
        tx (numpy.ndarray): Input data of shape (N, D).
        initial_w (numpy.ndarray): Initial guess for the model parameters of shape (D,).
        max_iters (int): Total number of iterations for GD.
        gamma (float): Step size (learning rate) for gradient updates.

    Returns:
        numpy.ndarray: Final weight vector of shape (D,).
        float: Final loss (MSE) value.
    """
    # Perform gradient descent to optimize the weights
    w, loss = gradient_descent(y, tx, initial_w, max_iters, gamma)

    # Return the final weights and the final loss
    return w, loss


### IMPLEMENTATION 2


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


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent (SGD) with MSE as the loss function.

    Args:
        y: numpy array of shape (N,), the target values.
        tx: numpy array of shape (N, D), the input data (N samples, D features).
        initial_w: numpy array of shape (D,), the initial guess for the model parameters.
        max_iters: integer, the number of iterations for SGD.
        gamma: scalar, the step size (learning rate) for gradient updates.

    Returns:
        w: numpy array of shape (D,), the final model parameters after SGD optimization.
        loss: scalar, the final loss (MSE) value.
    """

    # Perform stochastic gradient descent with a batch size of 1 (SGD)
    w, loss = stochastic_gradient_descent(y, tx, initial_w, 1, max_iters, gamma)

    # Return the final weights and loss
    return w, loss


### IMPLEMENTATION 3


def least_squares(y, tx):
    """Calculate the least squares solution using the normal equation.

    Args:
        y: numpy array of shape (N,), the target values.
        tx: numpy array of shape (N, D), the input data.

    Returns:
        w: numpy array of shape (D,), the optimal weights.
        mse: scalar, the mean squared error loss.

    >>> least_squares(np.array([0.1, 0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    # Compute the optimal weights using the normal equation: w = (X^T X)^{-1} X^T y
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)  # Using solve for stability

    # Compute the mean squared error (MSE) loss using the computed weights
    mse = compute_loss(y, tx, w)

    # Return the optimal weights and the MSE loss
    return w, mse


### IMPLEMENTATION 4


def ridge_regression(y, tx, lambda_):
    """Implement ridge regression using the normal equation.

    Args:
        y: numpy array of shape (N,), the target values.
        tx: numpy array of shape (N,D), the input data.
        lambda_: scalar, the regularization parameter.

    Returns:
        w: numpy array of shape (D,), the optimal weights.
        loss: scalar, the mean squared error loss.

    >>> ridge_regression(np.array([0.1, 0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    >>> ridge_regression(np.array([0.1, 0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    (array([0.03947092, 0.00319628]), 0.01600625)
    """
    # Number of data points
    N = len(y)

    # Adjust lambda for ridge regression
    lambda_prime = 2 * N * lambda_

    # Compute the optimal weights using the normal equation for ridge regression
    w = np.linalg.solve(tx.T @ tx + lambda_prime * np.identity(tx.shape[1]), tx.T @ y)

    # Compute the mean squared error (MSE) loss using the computed weights
    loss = compute_loss(y, tx, w)

    return w, loss


### IMPLEMENTATION 5


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


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Perform logistic regression using gradient descent.

    Args:
        y: numpy array of shape (N,), the target values (0 or 1).
        tx: numpy array of shape (N, D), the input data.
        initial_w: numpy array of shape (D,), the initial guess for the model parameters.
        max_iters: integer, the number of iterations for gradient descent.
        gamma: float, the step size (learning rate) for weight updates.

    Returns:
        w: numpy array of shape (D,), the final weight vector.
        loss: scalar, the final loss value.
    """

    # Initialize weights with the initial guess
    w = initial_w

    # Iterate over the number of iterations
    for iter in range(max_iters):
        # Perform one step of gradient descent
        w, loss_iter = learning_by_gradient_descent(y, tx, w, gamma)
        # Log info every 100 iterations
        if iter % 100 == 0:
            print(f"Current iteration={iter}, loss={loss_iter}")

    # Compute the final loss
    loss = calculate_loss_sigmoid(y, tx, w)

    return w, loss


### IMPLEMENTATION 6


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Perform regularized logistic regression using gradient descent.

    Args:
        y: numpy array of shape (N,), the target values (0 or 1).
        tx: numpy array of shape (N, D), the input data.
        lambda_: scalar, the regularization parameter (L2).
        initial_w: numpy array of shape (D,), the initial guess for the model parameters.
        max_iters: integer, the number of iterations for gradient descent.
        gamma: float, the step size (learning rate) for weight updates.

    Returns:
        w: The final weight vector.
        loss: The final loss value (without the regularization term).
    """
    w = initial_w  # Initialize weights

    for n_iter in range(max_iters):
        # Compute predicted probabilities (sigmoid)
        p = sigmoid(tx.dot(w))  # Shape (N,)

        # Compute the gradient of the loss
        gradient = tx.T.dot(p - y) / y.shape[0]  # Shape (D,)
        # Add regularization to the gradient
        gradient += 2 * lambda_ * w

        # Update the weights
        w -= gamma * gradient

        # Log info
        if n_iter % 100 == 0:
            print(
                f"Iteration {n_iter}/{max_iters}, loss={calculate_loss_sigmoid(y, tx, w)}, w={w}"
            )

    # Compute the final loss (without the regularization term)
    loss = calculate_loss_sigmoid(y, tx, w)

    # Return the final weights and loss (without regularization term)
    return w, loss
