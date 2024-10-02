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