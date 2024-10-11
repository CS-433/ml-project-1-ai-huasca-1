import os
from helpers import load_csv_data
import numpy as np
import matplotlib.pyplot as plt

# Loading the data
# data_path = os.path.join(os.getcwd(), 'dataset')
# print(data_path)
# x_train, x_test, y_train, train_ids, test_ids = load_csv_data(data_path)



### IMPLEMENTATION 1

def compute_loss(y, tx, w):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    return np.mean((y - tx.dot(w)) ** 2) / 2


def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    #return -(1/len(y)) * tx.T.dot(y-tx.dot(w))
    return -tx.T.dot(y - tx.dot(w)) / y.size


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD
    """
    ws, losses, w = [initial_w], [], initial_w
    for n_iter in range(max_iters):
        loss = compute_loss(y, tx, w)
        w -= gamma * compute_gradient(y, tx, w)

        ws.append(w)
        losses.append(loss)
        print(
            "GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )
    return losses, ws


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent (GD) with MSE as the loss function.

    Args:
        y: shape=(N, ). The target values.
        tx: shape=(N, D). The input data (N samples, D features).
        initial_w: shape=(D, ). The initial guess for the model parameters.
        max_iters: The total number of iterations for GD.
        gamma: The step size (learning rate).

    Returns:
        w: The final weight vector (after GD).
        loss: The final loss (MSE) value.
    """
    # Use the gradient_descent function to perform GD
    losses, ws = gradient_descent(y, tx, initial_w, max_iters, gamma)
    
    # Return the final weight and the final loss
    return ws[-1], losses[-1]



### IMPLEMENTATION 2

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.

    Example:

     Number of batches = 9

     Batch size = 7                              Remainder = 3
     v     v                                         v v
    |-------|-------|-------|-------|-------|-------|---|
        0       7       14      21      28      35   max batches = 6

    If shuffle is False, the returned batches are the ones started from the indexes:
    0, 7, 14, 21, 28, 35, 0, 7, 14

    If shuffle is True, the returned batches start in:
    7, 28, 14, 35, 14, 0, 21, 28, 7

    To prevent the remainder datapoints from ever being taken into account, each of the shuffled indexes is added a random amount
    8, 28, 16, 38, 14, 0, 22, 28, 9

    This way batches might overlap, but the returned batches are slightly more representative.

    Disclaimer: To keep this function simple, individual datapoints are not shuffled. For a more random result consider using a batch_size of 1.

    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)  # Number of data points.
    batch_size = min(data_size, batch_size)  # Limit the possible size of the batch.
    max_batches = int(data_size / batch_size)  # Maximum number of non-overlapping batches.

    # if shuffle:
    #     idxs = np.random.permutation(data_size)  # Shuffle indices
    # else:
    #     idxs = np.arange(data_size)

    # for start in range(0, data_size, batch_size):
    #     end = min(start + batch_size, data_size)
    #     yield y[idxs[start:end]], tx[idxs[start:end]]

    
    remainder = (
        data_size - max_batches * batch_size
    )  # Points that would be excluded if no overlap is allowed.

    if shuffle:
        # Generate an array of indexes indicating the start of each batch
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            # Add an random offset to the start of each batch to eventually consider the remainder points
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        # If no shuffle is done, the array of indexes is circular.
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        start_index = start  # The first data point of the batch
        end_index = (
            start_index + batch_size
        )  # The first data point of the following batch
        yield y[start_index:end_index], tx[start_index:end_index]


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from a data sample batch of size B, where B < N, and their corresponding labels.

    Args:
        y: numpy array of shape=(B, )
        tx: numpy array of shape=(B,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        A numpy array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """    
    return compute_gradient(y, tx, w)


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """
    ws, losses, w = [initial_w], [], initial_w

    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):

            w -= gamma * compute_stoch_gradient(minibatch_y, minibatch_tx, w)

            losses.append(compute_loss(minibatch_y, minibatch_tx, w))
            ws.append(w.copy())

        print(
            "SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=compute_loss(minibatch_y, minibatch_tx, w), w0=w[0], w1=w[1]
            )
        )
    return losses, ws


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent
    Args:
        y: numpy array of shape (N,), N is the number of samples (target values).
        tx: numpy array of shape (N, 2), N is the number of samples, 2 is the number of features.
        initial_w: numpy array of shape (2,), initial guess for the weights (parameters).
        max_iters: integer, the number of iterations to run the SGD loop.
        gamma: scalar, the step size (learning rate) for gradient updates.

    Returns:
        w: numpy array of shape (2,), the final model parameters after SGD optimization.
        losses: list of length max_iters, containing the loss values for each iteration of SGD.
    """
    # Call the stochastic_gradient_descent function to perform the SGD optimization
    losses, ws = stochastic_gradient_descent(y, tx, initial_w, batch_size=1, max_iters=max_iters, gamma=gamma)

    # Return the final weights (last entry in ws) and all the loss values
    return ws[-1], losses



### IMPLEMENTATION 3 

def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    # Compute the optimal weights using the normal equation: w = (X^T X)^{-1} X^T y
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))  
    # solves the system  (X^T * X )w = (X^T )y  without explicitly computing the inverse, 
    # which is more numerically stable than using np.linalg.inv.
  
    # returns optimal weights and mse
    return w, compute_loss(y, tx, w)



### IMPLEMENTATION 4

def ridge_regression(y, tx, lambda_):
    """Implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])
    """
    # Compute the optimal weights using the normal equation for ridge regression
    w = np.linalg.solve(tx.T @ tx + lambda_ * np.eye(tx.shape[1]), tx.T @ y)
    
    return w



### IMPLEMENTATION 5

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def calculate_hessian(y, tx, w):
    """Return the Hessian of the loss function.

    Args:
        y: shape=(N, 1)
        tx: shape=(N, D)
        w: shape=(D, 1)

    Returns:
        A Hessian matrix of shape=(D, D)
    """

    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]
    
    # Compute the predicted probabilities
    p = sigmoid(tx.dot(w)).flatten()  # Ensure p is a 1D array
    # Compute the diagonal elements of Hessian
    S = np.diag(p * (1 - p))  # Diagonal matrix S

    # Compute the Hessian matrix
    hessian = tx.T @ S @ tx  # X^T * S * X
    
    return hessian


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Perform logistic regression using gradient descent.

    Args:
        y: numpy array of shape (N,), the target values (0 or 1).
        tx: numpy array of shape (N, D), the input data.
        initial_w: numpy array of shape (D,), the initial guess for the model parameters.
        max_iters: integer, the number of iterations for gradient descent.
        gamma: float, the step size (learning rate) for weight updates.

    Returns:
        loss: scalar, the final loss value.
        gradient: numpy array of shape (D,), the final gradient.
        hessian: numpy array of shape (D, D), the final Hessian matrix.
    """
    
    w = initial_w  # Initialize weights
    for n_iter in range(max_iters):
        # Compute the predicted probabilities
        p = sigmoid(tx.dot(w))  # Shape (N,)
        
        # Calculate the loss
        loss = -np.mean(y * np.log(p + 1e-15) + (1 - y) * np.log(1 - p + 1e-15))
        
        # Compute the gradient
        gradient = tx.T.dot(p - y) / y.shape[0]  # Shape (D,)
        
        # Compute the Hessian
        hessian = calculate_hessian(y, tx, w)  # Shape (D, D)
        
        # Update weights using gradient descent
        w -= gamma * gradient
        
        # Optional: Print progress
        print(f"Iteration {n_iter + 1}/{max_iters}: loss={loss}, gradient={gradient.flatten()}, weights={w.flatten()}")
    
    return loss, gradient, hessian



