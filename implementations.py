from helpers_perso import *
import numpy as np
from helpers_perso.helpers_implementations import *


### IMPLEMENTATION 1 : Linear regression using gradient descent
def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Performs linear regression using gradient descent (GD) to minimize the Mean Squared Error (MSE) loss function.

    This function iteratively updates the model parameters `w` by calculating the gradient of the MSE loss with respect to 
    `w` and adjusting `w` in the opposite direction of the gradient, effectively minimizing the error over the dataset.

    Args:
        y (np.ndarray): Array of target values with shape (N,).
        tx (np.ndarray): Input data matrix with shape (N, D), where N is the number of samples and D is the number of features.
        initial_w (np.ndarray): Initial parameter vector with shape (D,), representing the starting values for the model parameters.
        max_iters (int): Number of iterations to run the gradient descent algorithm.
        gamma (float): Learning rate, controlling the size of each gradient update step.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Final parameter vector `w` of shape (D,) after convergence.
            - float: Final Mean Squared Error (MSE) value at the last iteration.

    Notes:
        - The function calls `gradient_descent`, which performs the iterative updates to optimize the parameters.
        - This function is suited for datasets of manageable size, where batch processing is feasible.
    """
    # Perform gradient descent to optimize the weights
    w, loss = gradient_descent(y, tx, initial_w, max_iters, gamma)

    # Return the final weights and the final loss
    return w, loss


### IMPLEMENTATION 2 : Linear regression using stochastic gradient descent
def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Performs linear regression using stochastic gradient descent (SGD) to minimize the Mean Squared Error (MSE) loss function.

    This function iteratively updates the model parameters `w` by computing gradients of the MSE loss with respect to each sample, effectively learning from one data point at a time. It is especially useful for large datasets where standard gradient descent would be computationally expensive.

    Args:
        y (numpy.ndarray): Array of shape (N,), containing the target values for each sample.
        tx (numpy.ndarray): Array of shape (N, D), representing the input data matrix with N samples and D features.
        initial_w (numpy.ndarray): Array of shape (D,), providing the initial values for the model parameters.
        max_iters (int): The total number of iterations to perform in the SGD process.
        gamma (float): The learning rate, controlling the step size for each parameter update.

    Returns:
        w (numpy.ndarray): Array of shape (D,), representing the optimized model parameters after completing SGD.
        loss (float): The final MSE value after the last iteration.

    Notes:
        - This function performs SGD with a batch size of 1, updating the model parameters on each individual sample.
        - The function calls `stochastic_gradient_descent`, which executes the iterative parameter updates.
    """
    # Perform stochastic gradient descent with a batch size of 1 (SGD)
    w, loss = stochastic_gradient_descent(y, tx, initial_w, 1, max_iters, gamma)

    # Return the final weights and loss
    return w, loss


### IMPLEMENTATION 3 : Least squares regression using normal equations
def least_squares(y, tx):
    """Computes the optimal weights for linear regression using the least squares method via the normal equations.

    This function solves for the weight vector `w` that minimizes the Mean Squared Error (MSE) loss, leveraging the 
    closed-form solution: `w = (X^T X)^(-1) X^T y`. This approach is efficient for small to medium-sized datasets 
    where inversion of `X^T X` is computationally feasible.

    Args:
        y (np.ndarray): Target values array of shape (N,), where N is the number of samples.
        tx (np.ndarray): Input data matrix of shape (N, D), where N is the number of samples and D is the number of features.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The optimal weight vector `w` of shape (D,).
            - float: The Mean Squared Error (MSE) of the model with the computed weights.

    Notes:
        - This method uses `np.linalg.solve` for numerical stability rather than directly inverting `X^T X`.
        - Suitable for datasets where `X^T X` is invertible; for high-dimensional data, consider regularization techniques to ensure stability.
    """
    # Compute the optimal weights using the normal equation: w = (X^T X)^{-1} X^T y
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)  # Using solve for stability

    # Compute the mean squared error (MSE) loss using the computed weights
    mse = compute_loss(y, tx, w)

    # Return the optimal weights and the MSE loss
    return w, mse


### IMPLEMENTATION 4 : Ridge regression using normal equations
def ridge_regression(y, tx, lambda_):
    """Performs ridge regression using the normal equation with L2 regularization.

    This function computes the optimal weights `w` by minimizing the Mean Squared Error (MSE) with an added 
    L2 regularization term, which helps prevent overfitting and stabilizes the solution. The normal equation 
    with regularization is given by: `w = (X^T X + lambda_prime * I)^(-1) X^T y`, where `lambda_prime` is 
    the adjusted regularization parameter.

    Args:
        y (np.ndarray): Target values array with shape (N,), where N is the number of samples.
        tx (np.ndarray): Input data matrix with shape (N, D), where D is the number of features.
        lambda_ (float): Regularization parameter (L2 penalty) that controls the amount of shrinkage applied to the weights.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The computed weight vector `w` of shape (D,).
            - float: The Mean Squared Error (MSE) with the optimized weights.

    Notes:
        - `lambda_` is adjusted internally as `lambda_prime = 2 * N * lambda_`, where `N` is the number of samples.
        - This function uses `np.linalg.solve` for numerical stability when solving the modified normal equation.
        - Ridge regression is ideal for multicollinear data, where standard least squares regression may be unstable.

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


### IMPLEMENTATION 5 : Logistic regression using gradient descent (y ∈ {0, 1})
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Performs binary logistic regression using gradient descent to minimize the logistic loss.

    This function optimizes the logistic regression model parameters `w` by iteratively adjusting them based on 
    the gradient of the logistic loss function. Logistic regression is useful for binary classification tasks, 
    where `y` represents class labels 0 or 1.

    Args:
        y (np.ndarray): Target values of shape (N,), where each entry is either 0 or 1, representing the two classes.
        tx (np.ndarray): Input data matrix of shape (N, D), where N is the number of samples and D is the number of features.
        initial_w (np.ndarray): Initial weights vector of shape (D,), providing the starting point for optimization.
        max_iters (int): Total number of gradient descent iterations to perform.
        gamma (float): Learning rate, which controls the step size of each gradient update.

    Returns:
        w: numpy array of shape (D,), the final weight vector.
        loss: scalar, the final loss value.
    """

    # Initialize weights with the initial guess
    w = initial_w

    # Ensure target values are correctly formatted for logistic regression
    y = mapping_log_reg(y)

    # Iterate over the number of iterations
    for iter in range(max_iters):
        # Perform one step of gradient descent
        w, loss_iter = learning_by_gradient_descent(y, tx, w, gamma)

        # Log info every iterations
        print(f"Current iteration={iter}, loss={loss_iter}")

    loss = calculate_loss_sigmoid(y, tx, w)

    return w, loss


### IMPLEMENTATION 6 : Regularized logistic regression using gradient descent (y ∈ {0, 1}, with regularization term λ∥w∥^2)
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Performs regularized logistic regression with L2 regularization using gradient descent.

    This function optimizes the logistic regression model parameters `w` by minimizing the regularized logistic loss, 
    which includes an L2 penalty term, λ∥w∥². The regularization term helps reduce overfitting by penalizing large weights, 
    making this method suitable for high-dimensional or sparse datasets.

    Args:
        y (np.ndarray): Target values array of shape (N,), where each entry is either 0 or 1, representing binary class labels.
        tx (np.ndarray): Input data matrix of shape (N, D), where N is the number of samples and D is the number of features.
        lambda_ (float): Regularization parameter (L2 penalty) that controls the amount of weight shrinkage applied.
        initial_w (np.ndarray): Initial parameter vector with shape (D,), used as the starting point for optimization.
        max_iters (int): Number of iterations to perform in the gradient descent algorithm.
        gamma (float): Learning rate, determining the step size for each gradient update.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The final optimized weight vector `w` of shape (D,).
            - float: The logistic loss (without regularization) evaluated with the final weights.

    Notes:
        - The regularization term, λ∥w∥², is added only to the gradient during updates; it is not included in the returned loss.
        - `mapping_log_reg(y)` is used to ensure target values are correctly formatted for logistic regression.
        - Intermediate progress is logged every 100 iterations for tracking convergence.
        - Suitable for binary classification problems where regularization helps improve generalization.
    """
    # Initialize weights with the initial guess
    w = initial_w

    # Ensure target values are correctly formatted for logistic regression
    y = mapping_log_reg(y)

    for n_iter in range(max_iters):
        # Compute predicted probabilities using the sigmoid function
        p = sigmoid(tx.dot(w))  # Shape (N,)

        # Compute the gradient of the regularized logistic loss
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
