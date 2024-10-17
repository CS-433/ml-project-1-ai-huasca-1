import os
from helpers import load_csv_data
import numpy as np
import matplotlib.pyplot as plt
import implementations as imp

# Loading the data
data_path = os.path.join(os.getcwd(), "dataset")
x_train, x_test, y_train, train_ids, test_ids = load_csv_data(data_path)
print("Data loaded successfully!")

remove_nan = True

def remove_nan_features(array):
    """
    Removes columns containing NaN values from a given array.

    Args:
        array (numpy.ndarray): The input array to clean.

    Returns:
        numpy.ndarray: The cleaned array with NaN-containing columns removed.
    """
    # Identify columns that contain NaN
    nan_cols = np.any(np.isnan(array), axis=0)

    # Remove columns containing NaN
    cleaned_array = array[:, ~nan_cols]

    return cleaned_array


# Clean all arrays by removing columns containing NaN values
x_train_cleaned = remove_nan_features(x_train)
x_test_cleaned = remove_nan_features(x_test)
print("Data cleaned successfully!")

# Print the shapes of the cleaned arrays to verify the column removal
print(f"Original shape of x_train: {x_train.shape}")
print(f"Cleaned shape of x_train: {x_train_cleaned.shape}")

print(f"Original shape of x_test: {x_test.shape}")
print(f"Cleaned shape of x_test: {x_test_cleaned.shape}")


# Initial weights (can be zeros, random, or some heuristic value)
initial_w = np.zeros(x_train_cleaned.shape[1])

# Parameters for gradient descent
max_iters = 1000
gamma = 0.1  # Learning rate


### Testing imprementation 1
print("Implementation 1: mean_squared_error_gd \n")

# Running gradient descent with MSE
w_final_gd, loss_final_gd = imp.mean_squared_error_gd(
    y_train.copy(), x_train_cleaned.copy(), initial_w.copy(), max_iters, gamma
)

print(f"Final weights: {w_final_gd}")
print(f"Final loss: {loss_final_gd}")


### Testing imprementation 2
print("Implementation 2: mean_squared_error_sgd \n")

batch_size = 1  # Size of mini-batch for stochastic updates

# Running stochastic gradient descent with MSE
w_final_sgd, loss_final_sgd = imp.mean_squared_error_sgd(
    y_train.copy(), x_train_cleaned.copy(), initial_w.copy(), max_iters, gamma
)

print(f"Final weights: {w_final_sgd}")
print(f"Final loss: {loss_final_sgd}")


### Testing implementation 3
print("Implementation 3: least_squares \n")

# Calculate the least squares solution
w_optimal, mse = imp.least_squares(y_train.copy(), x_train_cleaned.copy())

print(f"Optimal weights: {w_optimal}")
print(f"Mean Squared Error: {mse}")


### Testing implementation 4
print("Implementation 4: ridge_regression \n")

# Perform ridge regression
w_final_rr, loss_final_rr = imp.ridge_regression(
    y_train.copy(), x_train_cleaned.copy(), initial_w.copy(), max_iters, gamma
)

print(f"Final weights: {w_final_rr}")
print(f"Final loss: {loss_final_rr}")


### Testing implementation 5
print("Implementation 5: logistic_regression \n")

# Perform logistic regression
w_final_lr, loss_final_lr = imp.logistic_regression(
    y_train.copy(), x_train_cleaned.copy(), initial_w.copy(), max_iters, gamma
)

print(f"Final weights: {w_final_lr}")
print(f"Final loss: {loss_final_lr}")


### Testing implementation 6
print("Implementation 6: reg_logistic_regression \n")

# Example usage
lambda_ = 0.1  # Regularization parameter
gamma = 0.01  # Learning rate

# Perform regularized logistic regression
w_final_rlr, loss_final_rlr = imp.reg_logistic_regression(
    y_train.copy(), x_train_cleaned.copy(), lambda_, initial_w.copy(), max_iters, gamma
)

print(f"Final weights: {w_final_rlr}")
print(f"Final loss: {loss_final_rlr}")
