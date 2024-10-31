from helpers_perso import *
import numpy as np
from helpers_perso.helpers_implementations import *

### IMPLEMENTATION 1


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

    y = mapping_log_reg(y)

    # Iterate over the number of iterations
    for iter in range(max_iters):
        # Perform one step of gradient descent
        w, loss_iter = learning_by_gradient_descent(y, tx, w, gamma)

        # Log info every 100 iterations
        
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

    y = mapping_log_reg(y)

    for n_iter in range(max_iters):
        # Compute predicted probabilities (sigmoid)
        p = sigmoid(tx.dot(w))  # Shape (N,)

        # Compute the gradient of the loss
        gradient = tx.T.dot(p - y) / y.shape[0]  # Shape (D,)
        # Add regularization to the gradient
        gradient += 2 * lambda_ * w

        # Update the weights
        w -= gamma * gradient

        # Log info every 100 iterations
        if n_iter % 100 == 0:
            print(f"Iteration {n_iter}/{max_iters}, loss={calculate_loss_sigmoid(y, tx, w)}, w={w}")

    # Compute the final loss (without the regularization term)
    loss = calculate_loss_sigmoid(y, tx, w)

    # Return the final weights and loss (without regularization term)
    return w, loss
