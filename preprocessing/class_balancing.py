import numpy as np

def balance_classes(X, y, ratio=1):
    # Identify indices of each class
    class1_ids = np.where(y == -1)[0]
    class2_ids = np.where(y == 1)[0]

    # Determine majority and minority class indices
    majority_class_ids = class1_ids if len(class1_ids) > len(class2_ids) else class2_ids
    minority_class_ids = class1_ids if len(class1_ids) < len(class2_ids) else class2_ids

    # Calculate NaN proportion for each row in X
    nan_proportion = np.isnan(X).mean(axis=1)

    # Get the majority class indices sorted by highest NaN proportion
    majority_class_nan_sorted = majority_class_ids[np.argsort(-nan_proportion[majority_class_ids])]

    # Determine the number of majority samples to keep based on the ratio
    target_majority_count = int(len(minority_class_ids) * ratio)

    # Select the subset of majority samples to retain
    selected_majority_class_ids = majority_class_nan_sorted[:target_majority_count]

    # Get the indices of deleted rows
    deleted_ids = np.setdiff1d(majority_class_ids, selected_majority_class_ids)

    # Combine the indices of the retained majority class samples with the minority class samples
    balanced_ids = np.concatenate([selected_majority_class_ids, minority_class_ids])

    # Return the balanced dataset and the deleted indices
    return X[balanced_ids], y[balanced_ids], deleted_ids
