import os
from helpers import load_csv_data
import numpy as np
import matplotlib.pyplot as plt
import implementations as imp
from helpers_perso import *
from crossvalidation import *

# Loading the data
data_path = os.path.join(os.getcwd(), "data", "dataset")
x_train, x_test, y_train, train_ids, test_ids = load_csv_data(data_path)
print("Data loaded successfully!")

# # Initial weights (can be zeros, random, or some heuristic value)
# initial_w = np.zeros(x_train.shape[1])

# # Parameters for gradient descent
# max_iters = 1000
# gamma = 0.1  # Learning rate


# ### Testing imprementation 1
# print("Implementation 1: mean_squared_error_gd \n")

# # Running gradient descent with MSE
# w_final_gd, loss_final_gd = imp.mean_squared_error_gd(
#     y_train.copy(), x_train_cleaned.copy(), initial_w.copy(), max_iters, gamma
# )

# print(f"Final weights: {w_final_gd}")
# print(f"Final loss: {loss_final_gd}")


# ### Testing imprementation 2
# print("Implementation 2: mean_squared_error_sgd \n")

# batch_size = 1  # Size of mini-batch for stochastic updates

# # Running stochastic gradient descent with MSE
# w_final_sgd, loss_final_sgd = imp.mean_squared_error_sgd(
#     y_train.copy(), x_train_cleaned.copy(), initial_w.copy(), max_iters, gamma
# )

# print(f"Final weights: {w_final_sgd}")
# print(f"Final loss: {loss_final_sgd}")


# ### Testing implementation 3
# print("Implementation 3: least_squares \n")

# # Calculate the least squares solution
# w_optimal, mse = imp.least_squares(y_train.copy(), x_train_cleaned.copy())

# print(f"Optimal weights: {w_optimal}")
# print(f"Mean Squared Error: {mse}")


# ### Testing implementation 4
# print("Implementation 4: ridge_regression \n")

# # Perform ridge regression
# w_final_rr, loss_final_rr = imp.ridge_regression(
#     y_train.copy(), x_train_cleaned.copy(), initial_w.copy(), max_iters, gamma
# )

# print(f"Final weights: {w_final_rr}")
# print(f"Final loss: {loss_final_rr}")


# ### Testing implementation 5
# print("Implementation 5: logistic_regression \n")

# # Perform logistic regression
# w_final_lr, loss_final_lr = imp.logistic_regression(
#     y_train.copy(), x_train_cleaned.copy(), initial_w.copy(), max_iters, gamma
# )

# print(f"Final weights: {w_final_lr}")
# print(f"Final loss: {loss_final_lr}")


# ### Testing implementation 6
# print("Implementation 6: reg_logistic_regression \n")

# # Example usage
# lambda_ = 0.1  # Regularization parameter
# gamma = 0.01  # Learning rate

# # Perform regularized logistic regression
# w_final_rlr, loss_final_rlr = imp.reg_logistic_regression(
#     y_train.copy(), x_train_cleaned.copy(), lambda_, initial_w.copy(), max_iters, gamma
# )

# print(f"Final weights: {w_final_rlr}")
# print(f"Final loss: {loss_final_rlr}")

# import os
# os.chdir("..")

from helpers import *
from helpers_perso import *
from preprocessing import nan_imputation
from preprocessing import one_hot_encoding
from preprocessing import standardization

# Clean all arrays by removing columns containing NaN values
x_train_cleaned = nan_imputation.remove_nan_features(x_train, 0.8)
print(
    f"Removed {x_train.shape[1] - x_train_cleaned.shape[1]} columns with more than 80% NaN values."
)

# Identify columns that only contain integers
integer_columns = [
    i
    for i in range(x_train_cleaned.shape[1])
    if np.all(np.mod(x_train_cleaned[:, i][~np.isnan(x_train_cleaned[:, i])], 1) == 0)
]
non_integer_columns = [
    i for i in range(x_train_cleaned.shape[1]) if i not in integer_columns
]
assert len(integer_columns) + len(non_integer_columns) == x_train_cleaned.shape[1]

x_train_cleaned_without_nans = nan_imputation.encode_nan_integer_columns(
    x_train_cleaned, replacement_value="mode"
)
x_train_cleaned_without_nans = nan_imputation.encode_nan_continuous_columns(
    x_train_cleaned_without_nans, replacement_value="mode"
)
print("NaN values encoded successfully!")

assert np.isnan(x_train_cleaned_without_nans).sum() == 0
assert x_train_cleaned.shape == x_train_cleaned_without_nans.shape

# Calculate the number of unique values for each integer-only column
unique_value_counts = np.array(
    [len(np.unique(x_train_cleaned[:, col])) for col in integer_columns]
)

# Define columns to One-Hot-Encode based on the number of unique values
categorical_treshold = 5
indexes_categorical_features = [
    i for i, count in enumerate(unique_value_counts) if count <= categorical_treshold
]
# Find indexes of non-categorical features that are not in the categorical features list
indexes_non_categorical_features = [
    i for i in range(len(unique_value_counts)) if i not in indexes_categorical_features
]
assert len(indexes_categorical_features) + len(indexes_non_categorical_features) == len(
    unique_value_counts
)
assert unique_value_counts.size == len(integer_columns)


encoded_x_train = one_hot_encoding.binary_encode_columns(
    x_train_cleaned_without_nans, indexes_categorical_features
)

standardized_x_train = standardization.standardize_columns(
    encoded_x_train, indexes_non_categorical_features
)

###### a partir de la peut etre pas opti mais standardized_x_train est standardisé et sans Nan....
