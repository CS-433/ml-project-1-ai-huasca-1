import numpy as np
from implementations import *
from helpers_perso import *


def cross_validate(model_fn, X, y, initial_weights=None, k=5, **kwargs):
    """
    Perform k-Fold Cross-Validation on a specified model function to evaluate its performance.

    This function divides the data into `k` folds, trains the model on `k-1` folds, and validates it on the remaining fold, 
    iteratively across all folds. It returns the average validation loss, providing an overall estimate of model performance 
    and reducing variance associated with a single train/test split.

    Args:
        model_fn (callable): Model function from `implementations.py` that trains the model and returns weights and a loss.
        X (np.ndarray): Input feature matrix of shape (N, D), where N is the number of samples and D is the number of features.
        y (np.ndarray): Target values array of shape (N,).
        initial_weights (np.ndarray, optional): Initial weights array to be used in models that require it, of shape (D,).
        k (int, optional): Number of folds for cross-validation; defaults to 5.
        **kwargs: Additional keyword arguments to be passed to `model_fn`.

    Returns:
        float: The average validation loss over all k folds.

    Notes:
        - The model function should be compatible with inputs `(y_train, X_train, initial_w, **kwargs)`.
        - For loss calculation, this function uses `compute_loss`, which should be defined (e.g., Mean Squared Error for regression).
        - For models not requiring initial weights, `initial_weights` can be left as `None`.

    """
    # Shuffle the data indices for random folding
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    
    # Split indices into k approximately equal folds
    fold_size = len(y) // k
    folds = [indices[i * fold_size : (i + 1) * fold_size] for i in range(k)]
    
    # Handle any remaining samples by adding them to the last fold
    if len(y) % k != 0:
        folds[-1] = np.concatenate((folds[-1], indices[k * fold_size:]))

    losses = []
    
    # Perform k-fold cross-validation
    for i in range(k):
        # Define validation fold
        val_indices = folds[i]
        
        # Define training folds by excluding the current validation fold
        train_indices = np.concatenate([folds[j] for j in range(k) if j != i])
        
        # Prepare training and validation data
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        
        # Call the model function with or without initial_weights
        if "initial_w" in model_fn.__code__.co_varnames:
            w, loss = model_fn(y_train, X_train, initial_w=initial_weights, **kwargs)
        else:
            w, loss = model_fn(y_train, X_train, **kwargs)
        
        # Compute the validation loss and store it
        val_loss = compute_loss(y_val, X_val, w)  # Assuming compute_loss is MSE or another loss function
        losses.append(val_loss)

    # Return the average loss across all folds
    return np.mean(losses)


def tune_hyperparameters(models, param_grid, X, y, initial_weights, k=5):
    """
    Tune hyperparameters using grid search and k-fold cross-validation.

    Args:
        models (dict): Dictionary with model names as keys and functions as values.
        param_grid (dict): Dictionary of parameter grids for each model.
        X (numpy.ndarray): Input features.
        y (numpy.ndarray): Target values.
        initial_weights (numpy.ndarray): Initial weights to use for all models.
        k (int): Number of folds for cross-validation.

    Returns:
        dict: Best hyperparameters and scores for each model.
    """
    tuning_results = {}

    for model_name, model_fn in models.items():
        print(f"\nTuning {model_name}...")

        model_params = param_grid[model_name]
        best_score = float('inf')
        best_params = None

        def recursive_grid_search(depth, current_params):
            nonlocal best_score, best_params
            param_names = list(model_params.keys())
            param_values = list(model_params.values())
            
            # If at the last depth, run cross-validation
            if depth == len(param_names):
                params = dict(current_params)
                
                # Perform cross-validation with the current parameter combination
                cv_score = cross_validate(model_fn, X=X, y=y, initial_weights=initial_weights, k=k, **params)
                print(f"Params: {params}, Cross-validated loss: {cv_score}")

                # Check if the current score is better than the best found so far
                if cv_score < best_score:
                    best_score = cv_score
                    best_params = params

                return

            # Recursively search over the parameter values
            for value in param_values[depth]:
                recursive_grid_search(depth + 1, current_params + [(param_names[depth], value)])

        # Start the recursive grid search
        recursive_grid_search(0, [])
        
        # Store the best result for the current model
        tuning_results[model_name] = {"best_params": best_params, "best_score": best_score}
        print(f"Best for {model_name}: Params = {best_params}, Score = {best_score}\n")

    return tuning_results
