import os
from helpers import load_csv_data, create_csv_submission
import numpy as np
import matplotlib.pyplot as plt
from implementations import *
from helpers_perso import *
from crossvalidation import *
from preprocessing.nan_imputation import *
from preprocessing.binary_encoding import *
from preprocessing.standardization import *
from prediction_and_evaluation.predict_labels import (
    predict_classification,
    predict_classification_logistic,
)
from preprocessing.class_balancing import balance_classes
from preprocessing.remove_highly_correlated_features import (
    remove_highly_correlated_features,
)

# Loading the data
data_path = os.path.join(os.getcwd(), "data", "dataset")
x_train, x_test, y_train, train_ids, test_ids = load_csv_data(data_path)
print("Data loaded successfully!")

# Balancing the data (ratio of balancing_ratio between majority and minority classes)
balancing_ratio = 1
x_balanced, y_balanced, deleted_ids = balance_classes(x_train, y_train, balancing_ratio)
print(f"Classes balanced successfully! : ratio {balancing_ratio}")

# Removing features containing a proportion of NaN values greater than nan_proportion_to_remove
nan_proportion_to_remove = 0.3
x_train_cleaned, deleted_indices = remove_nan_features(
    x_balanced, nan_proportion_to_remove
)
adapted_x_test = np.delete(x_test, deleted_indices, axis=1)
print(f"NaN features removed successfully! : proportion {nan_proportion_to_remove}")

# Identifying integer and non-integer columns
integer_columns, non_integer_columns = identify_integer_columns(x_train_cleaned)
assert len(integer_columns) + len(non_integer_columns) == x_train_cleaned.shape[1]

# Nan imputation as mode value
x_train_cleaned_without_nans = encode_nan_integer_columns(
    x_train_cleaned, replacement_value="mode"
)
x_train_cleaned_without_nans = encode_nan_continuous_columns(
    x_train_cleaned_without_nans, replacement_value="mode"
)
assert np.isnan(x_train_cleaned_without_nans).sum() == 0
assert x_train_cleaned.shape == x_train_cleaned_without_nans.shape
adapted_x_test_without_nans = encode_nan_integer_columns(
    adapted_x_test, replacement_value="mode"
)
adapted_x_test_without_nans = encode_nan_continuous_columns(
    adapted_x_test_without_nans, replacement_value="mode"
)
assert np.isnan(adapted_x_test_without_nans).sum() == 0
assert adapted_x_test.shape == adapted_x_test_without_nans.shape
print("Nan values encoded successfully as the mode!")

# Identifying categorical and non-categorical features (number of unique values over categorical_treshold -> quantitative)
categorical_threshold = 5
unique_value_counts = np.array(
    [len(np.unique(x_train_cleaned[:, col])) for col in integer_columns]
)
indexes_categorical_features = [
    integer_columns[i]
    for i, count in enumerate(unique_value_counts)
    if count <= categorical_threshold
]
indexes_non_categorical_features = [
    integer_columns[i]
    for i in range(len(unique_value_counts))
    if integer_columns[i] not in indexes_categorical_features
]
assert len(indexes_categorical_features) + len(indexes_non_categorical_features) == len(
    unique_value_counts
)
assert unique_value_counts.size == len(integer_columns)
indexes_non_categorical_features.extend(non_integer_columns)

# Standardizing the non-categorical features
x_standardized = standardize_columns(
    x_train_cleaned_without_nans, indexes_non_categorical_features
)
x_test_standardized = standardize_columns(
    adapted_x_test_without_nans, indexes_non_categorical_features
)

# Binary encoding the categorical features
encoded_x_train, encoded_x_test = consistent_binary_encode(
    x_standardized, x_test_standardized, indexes_categorical_features
)

# Initialising the weights and hyperparameters according to optimal values
initial_w = np.zeros(encoded_x_train.shape[1])
max_iters = 1000
gamma = 0.3
lambda_ = 0.1

# Run the model
w, loss = logistic_regression(
    y_balanced, encoded_x_train, initial_w, max_iters, gamma
)

# Predict the labels
y_test = predict_classification_logistic(encoded_x_test, w)

# Writing the submission file
create_csv_submission(test_ids, y_test, "submission_log_reg.csv")
