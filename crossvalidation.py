import numpy as np
from implementations import *

def cross_validate(model_fn, X, y, k=5, **kwargs):
    """
    Perform k-Fold Cross-Validation on a given model.

    Args:
        model_fn (function): The model function from implementations.py to use.
        X (numpy.ndarray): Input features of shape (N, D).
        y (numpy.ndarray): Target values of shape (N,).
        k (int): Number of folds for cross-validation.
        **kwargs: Additional arguments for the model function, it's a dynamical argument.

    Returns:
        float: The average loss across all k folds.
    """
    # Shuffle the data indices --> ensure each fold has representative distribution 
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    
    # Split indices into k folds
    fold_size = len(y) // k
    folds = [indices[i * fold_size : (i + 1) * fold_size] for i in range(k)]

    losses = []
    
    # Select training and validation indices
    for i in range(k):
        # Define the validation fold
        val_indices = folds[i]
        
        # Define the training folds by excluding the current fold
        train_indices = np.concatenate([folds[j] for j in range(k) if j != i])
        
        # Prepare train and validation data
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        
        # Call the provided model function and pass any additional arguments 
        # Flexible Model function : any function of implementations.py can be used
        # w, loss = model_fn(y_train, X_train, **kwargs)
        w,loss = model_fn(y_train, X_train, **kwargs)
        
        # Evaluate model on validation set and save the score
        val_loss = compute_loss(y_val, X_val, w)  # Assuming compute_loss is MSE or other appropriate loss
        losses.append(val_loss)

    # Return the average loss across all folds
    return np.mean(losses)