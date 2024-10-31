import numpy as np

def remove_highly_correlated_features(X, threshold=0.9):
    """
    Remove features from X that are highly correlated with each other. In fact it can cause instability in models that relies
    on matrix inversion (least_squares and ridge_regression)
    
    Args:
        X (np.ndarray): The feature matrix of shape (n_samples, n_features).
        threshold (float): The correlation threshold above which a feature will be removed.
    
    Returns:
        np.ndarray: The reduced feature matrix with highly correlated features removed.
        list: Indices of the features that were removed.
    """
    # Calculate the pearson correlation matrix, measure of the linear relationship between features ranging from -1 to 1
    correlation_matrix = np.corrcoef(X, rowvar=False) # The correlation matrix reveals which features are linearly dependent, helping identify redundant information.
    #rowvar=False, NumPy treats columns as variables and rows as observations. 
    # because each column represents a different feature and each row is an individual observation.
    correlation_matrix = np.abs(correlation_matrix) # take absolute value, so both positive and negative correlation are considered equaly
    
    # Identify indices of features to drop
    to_drop = set()
    for i in range(correlation_matrix.shape[0]):
        for j in range(i + 1, correlation_matrix.shape[1]):
            if correlation_matrix[i, j] > threshold: # remove if correlation is above threshold
                to_drop.add(j)  # Add the index of one feature from each correlated pair

    # Remove correlated features from X
    X_reduced = np.delete(X, list(to_drop), axis=1)
    
    print(f"Removed {len(to_drop)} features due to high correlation.")
    return X_reduced, list(to_drop)