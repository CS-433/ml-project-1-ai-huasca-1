import numpy as np
from predict_labels import predict_classification

def compute_scores(y, x, w):
    """
    Compute the accuracy and F1 score for each data point in the dataset.

    Args:
        y (np.ndarray): The target values of shape (n_samples,).
        x (np.ndarray): The input data of shape (n_samples, n_features).
        w (np.ndarray): The weight vector of shape (n_features,).

    Returns:
        tuple: A tuple containing accuracy and F1 score.
    """
    # Predict binary classifications based on the weights
    predictions = predict_classification(x, w)

    # Calculate accuracy
    accuracy = np.sum(predictions == y) / len(y)

    # Calculate F1 score components
    true_positives = np.sum((predictions == 1) & (y == 1))
    false_positives = np.sum((predictions == 1) & (y == -1))
    false_negatives = np.sum((predictions == -1) & (y == 1))
    true_negatives = np.sum((predictions == -1) & (y == -1))

    # Check that all cases add up to the total sample size
    assert true_positives + false_positives + false_negatives + true_negatives == len(y)

    # Avoid division by zero for F1 score
    if true_positives + false_positives == 0 or true_positives + false_negatives == 0:
        f1_score = 0
    else:
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1_score = 2 * (precision * recall) / (precision + recall)

    return accuracy, f1_score
