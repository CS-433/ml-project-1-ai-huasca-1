import os
from helpers import load_csv_data, create_csv_submission
import numpy as np
import matplotlib.pyplot as plt
from implementations import *
from helpers_perso import *
from crossvalidation import *
from preprocessing.nan_imputation import *
from preprocessing.one_hot_encoding import *
from preprocessing.standardization import *
from predict_labels import predict_classification
from preprocessing.class_balancing import balance_classes


# Loading the data
data_path = os.path.join(os.getcwd(), "data", "dataset")
x_train, x_test, y_train, train_ids, test_ids = load_csv_data(data_path)
print("Data loaded successfully!")

x_balanced, y_balanced, deleted_ids = balance_classes(x_train, y_train, 1)

x_train_cleaned, deleted_indices = remove_nan_features(x_balanced, 0.8)
adapted_x_test = np.delete(x_test, deleted_indices, axis=1)


integer_columns, non_integer_columns = identify_integer_columns(x_train_cleaned)
assert len(integer_columns) + len(non_integer_columns) == x_train_cleaned.shape[1]

x_train_cleaned_without_nans = encode_nan_integer_columns(x_train_cleaned, replacement_value='mode')
x_train_cleaned_without_nans = encode_nan_continuous_columns(x_train_cleaned_without_nans, replacement_value='mode')
assert np.isnan(x_train_cleaned_without_nans).sum() == 0
assert x_train_cleaned.shape == x_train_cleaned_without_nans.shape
adapted_x_test_without_nans = encode_nan_integer_columns(adapted_x_test, replacement_value='mode')
adapted_x_test_without_nans = encode_nan_continuous_columns(adapted_x_test_without_nans, replacement_value='mode')
assert np.isnan(adapted_x_test_without_nans).sum() == 0
assert adapted_x_test.shape == adapted_x_test_without_nans.shape

categorical_threshold = 5
unique_value_counts = np.array([len(np.unique(x_train_cleaned[:, col])) for col in integer_columns])
indexes_categorical_features = [integer_columns[i] for i, count in enumerate(unique_value_counts) if count <= categorical_threshold]
indexes_non_categorical_features = [integer_columns[i] for i in range(len(unique_value_counts)) if integer_columns[i] not in indexes_categorical_features]
assert len(indexes_categorical_features) + len(indexes_non_categorical_features) == len(unique_value_counts)
assert unique_value_counts.size == len(integer_columns)
indexes_non_categorical_features.extend(non_integer_columns)

x_standardized = standardize_columns(x_train_cleaned_without_nans, indexes_non_categorical_features)
x_test_standardized = standardize_columns(adapted_x_test_without_nans, indexes_non_categorical_features)

encoded_x_train, encoded_x_test = consistent_binary_encode(x_standardized, x_test_standardized, indexes_categorical_features)

initial_w = np.zeros(encoded_x_train.shape[1])
max_iters = 150
gamma = 0.001

w, loss = mean_squared_error_gd(y_balanced, encoded_x_train, initial_w, max_iters, gamma)

y_test = predict_classification(encoded_x_test,w)

create_csv_submission(test_ids, y_test, "submission.csv")






# # Initial weights (can be zeros, random, or some heuristic value)
# initial_w = np.zeros(x_train.shape[1])

# # Parameters for gradient descent
# max_iters = 1000
# gamma = 0.1  # Learning rate


# ### Testing imprementation 1
# print("Implementation 1: mean_squared_error_gd \n")

# # Running gradient descent with MSE
# w_final_gd, loss_final_gd = mean_squared_error_gd(
#     y_train.copy(), x_train_cleaned.copy(), initial_w.copy(), max_iters, gamma
# )

# print(f"Final weights: {w_final_gd}")
# print(f"Final loss: {loss_final_gd}")


# ### Testing imprementation 2
# print("Implementation 2: mean_squared_error_sgd \n")

# batch_size = 1  # Size of mini-batch for stochastic updates

# # Running stochastic gradient descent with MSE
# w_final_sgd, loss_final_sgd = mean_squared_error_sgd(
#     y_train.copy(), x_train_cleaned.copy(), initial_w.copy(), max_iters, gamma
# )

# print(f"Final weights: {w_final_sgd}")
# print(f"Final loss: {loss_final_sgd}")


# ### Testing implementation 3
# print("Implementation 3: least_squares \n")

# # Calculate the least squares solution
# w_optimal, mse = least_squares(y_train.copy(), x_train_cleaned.copy())

# print(f"Optimal weights: {w_optimal}")
# print(f"Mean Squared Error: {mse}")


# ### Testing implementation 4
# print("Implementation 4: ridge_regression \n")

# # Perform ridge regression
# w_final_rr, loss_final_rr = ridge_regression(
#     y_train.copy(), x_train_cleaned.copy(), initial_w.copy(), max_iters, gamma
# )

# print(f"Final weights: {w_final_rr}")
# print(f"Final loss: {loss_final_rr}")


# ### Testing implementation 5
# print("Implementation 5: logistic_regression \n")

# # Perform logistic regression
# w_final_lr, loss_final_lr = logistic_regression(
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
# w_final_rlr, loss_final_rlr = reg_logistic_regression(
#     y_train.copy(), x_train_cleaned.copy(), lambda_, initial_w.copy(), max_iters, gamma
# )

# print(f"Final weights: {w_final_rlr}")
# print(f"Final loss: {loss_final_rlr}")

# import os
# os.chdir("..")

