import os
from helpers import load_csv_data
import numpy as np

# Loading the data
data_path = os.path.join(os.getcwd(), 'dataset')
x_train, x_test, y_train, train_ids, test_ids = load_csv_data(data_path)

# print(f"x_train shape: {x_train.shape}")
# print(f"y_train shape: {y_train.shape}")
# print(f"x_test shape: {x_test.shape}")


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
    # Initialize the weights
    w = initial_w
    
    # Perform gradient descent for max_iters iterations
    for n_iter in range(max_iters):
        # Compute the prediction error (residual)
        e = y - tx.dot(w)
        
        # Compute the loss (MSE)
        loss = np.mean(e ** 2) / 2
        
        # Compute the gradient of the loss
        grad = -tx.T.dot(e) / y.size
        
        # Update the weights using gradient descent step
        w = w - gamma * grad
        
        # Optionally, print the progress for each iteration
        print(f"Iteration {n_iter+1}/{max_iters}: loss={loss}, w={w}")
    
    # Return the final weight vector and final loss
    return w, loss

#Importations
import numpy as np
import matplotlib.pyplot as plt

###IMPLEMENTATION 2

def compute_loss(y, tx, w):
    """Calculate the loss using MSE.

    Args:
        y: numpy array of shape (N,)
        tx: numpy array of shape (N, 2)
        w: numpy array of shape (2,). The vector of model parameters.

    Returns:
        mse: scalar, the value of the loss (MSE) corresponding to the input parameters w.
    """
    # Compute the predictions (Xw)
    predictions = tx.dot(w)
    
    # Compute the error vector (e = y - Xw)
    e = y - predictions

    # Compute the Mean Squared Error (MSE)
    mse = (1 / (2 * len(y))) * np.dot(e.T, e)

    return mse

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
    data_size = len(y)  # NUmber of data points.
    batch_size = min(data_size, batch_size)  # Limit the possible size of the batch.
    max_batches = int(
        data_size / batch_size
    )  # The maximum amount of non-overlapping batches that can be extracted from the data.
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
    # Stochastic gradient computation. It's the same as the usual gradient.
    # Compute the error vector for the mini-batch
    e = y - tx.dot(w)
    
    # Compute the gradient based on the mini-batch
    gradient = - (1 / len(y)) * tx.T.dot(e)
    
    return gradient

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

    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: implement stochastic gradient descent.
        # ***************************************************
        # Get the next mini-batch using batch_iter() function (assuming it generates mini-batches)
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            # Compute the stochastic gradient using the mini-batch
            gradient = compute_stoch_gradient(minibatch_y, minibatch_tx, w)
            
            # Update weights using the gradient and step size (gamma)
            w = w - gamma * gradient
            
            # Compute the loss using Mean Squared Error (MSE) for the current weights
            loss = compute_loss(y, tx, w)
            
            # Store the updated weights and loss
            ws.append(w)
            losses.append(loss)

            print(
                "SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                    bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
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


##Implementation 3 

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

    # returns mse, and optimal weights
    # Compute the optimal weights using the normal equation: w = (X^T X)^{-1} X^T y
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))  
    #solves the system  (X^T * X )w = (X^T )y  without explicitly computing the inverse, 
    # which is more numerically stable than using np.linalg.inv.
    
    # Compute the mean squared error (MSE)
    e = y - tx.dot(w)  # Error vector
    mse = (1 / (2 * len(y))) * np.dot(e.T, e)
  
    return w, mse



# Implementation 5

def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute gradient vector
    predictions = tx.dot(w)
    
    # Compute the error vector (e = y - Xw)
    e = y - predictions
    N = len(y)
    
    # Compute the gradient
    gradient = -(1/N) * tx.T.dot(e)
    # ***************************************************
    #raise NotImplementedError
    return gradient

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
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []

    # Start with the initial weights
    w = initial_w

    #Gradient descent loop
    for n_iter in range(max_iters):
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: compute gradient and loss
        # Compute the gradient and the loss
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        # ***************************************************
        #raise NotImplementedError
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: update w by gradient
        # Update the weights using the gradient and step size gamma
        w = w - gamma * gradient
        # ***************************************************
        #raise NotImplementedError

        # store w and loss
        ws.append(w)
        losses.append(loss)
        print(
            "GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )

    return losses, ws


#def logistic_regression(y, tx, initial_w, max_iters, gamma):