import numpy as np
from implementations import *
from helpers_perso import *


import numpy as np

def cross_validate(model_fn, X, y, initial_weights=None, k=5, **kwargs):
    """
    Perform k-Fold Cross-Validation on a given model.

    Args:
        model_fn (function): The model function from implementations.py to use.
        X (numpy.ndarray): Input features of shape (N, D).
        y (numpy.ndarray): Target values of shape (N,).
        initial_weights (numpy.ndarray): Initial weights to use for models that need it.
        k (int): Number of folds for cross-validation.
        **kwargs: Additional arguments for the model function.

    Returns:
        float: The average loss across all k folds.
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


# def cross_validate(model_fn, X, y, initial_weights=None, k=5, **kwargs):
#     """
#     Perform k-Fold Cross-Validation on a given model.

#     Args:
#         model_fn (function): The model function from implementations.py to use.
#         X (numpy.ndarray): Input features of shape (N, D).
#         y (numpy.ndarray): Target values of shape (N,).
#         initial_weights (numpy.ndarray): Initial weights to use for models that need it.
#         k (int): Number of folds for cross-validation.
#         **kwargs: Additional arguments for the model function.

#     Returns:
#         float: The average loss across all k folds.
#     """
#     # Shuffle the data indices to ensure each fold has a representative distribution 
#     indices = np.arange(len(y))
#     np.random.shuffle(indices)
    
#     # Split indices into k folds
#     fold_size = len(y) // k
#     folds = [indices[i * fold_size : (i + 1) * fold_size] for i in range(k)]

#     losses = []
    
#     # Select training and validation indices
#     for i in range(k):
#         # Define the validation fold
#         val_indices = folds[i]
        
#         # Define the training folds by excluding the current fold
#         train_indices = np.concatenate([folds[j] for j in range(k) if j != i])
        
#         # Prepare train and validation data
#         X_train, y_train = X[train_indices], y[train_indices]
#         X_val, y_val = X[val_indices], y[val_indices]
        
#         # Call the model function with initial_weights as initial_w
#         w, loss = model_fn(y_train, X_train, initial_w=initial_weights, **kwargs)
        
#         # Evaluate model on validation set and save the score
#         val_loss = compute_loss(y_val, X_val, w)  # Assuming compute_loss is MSE or other appropriate loss
#         losses.append(val_loss)

#     # Return the average loss across all folds
#     return np.mean(losses)


# def tune_hyperparameters(models, param_grid, X, y, initial_weights, k=5):
#     """
#     Tune hyperparameters using grid search and k-fold cross-validation.

#     Args:
#         models (dict): Dictionary with model names as keys and functions as values.
#         param_grid (dict): Dictionary of parameter grids for each model.
#         X (np.ndarray): Input features.
#         y (np.ndarray): Target values.
#         initial_weights (np.ndarray): Initial weights to use for models that need it.
#         k (int): Number of folds for cross-validation.

#     Returns:
#         dict: Best hyperparameters and scores for each model.
#     """
#     tuning_results = {}

#     for model_name, model_fn in models.items():
#         print(f"\nCross-validating and tuning {model_name}...")

#         model_params = param_grid[model_name]
#         best_score = float('inf')
#         best_params = None
#         iteration = 0

#         def grid_search_recursive(depth, current_params):
#             nonlocal best_score, best_params, iteration
#             param_names = list(model_params.keys())
#             param_values = list(model_params.values())

#             if depth == len(param_names):
#                 params = dict(current_params)
                
#                 # Conditionally include initial_weights only for models that need it
#                 if model_name in ["ridge_regression", "least_squares"]:
#                     cv_score = cross_validate(
#                         model_fn,
#                         X=X,
#                         y=y,
#                         k=k,
#                         **params
#                     )
#                 else:
#                     cv_score = cross_validate(
#                         model_fn,
#                         X=X,
#                         y=y,
#                         initial_weights=initial_weights,
#                         k=k,
#                         **params
#                     )
                
#                 print(f"Iteration {iteration}: Params: {params}, Cross-validated loss: {cv_score}")

#                 if cv_score < best_score:
#                     best_score = cv_score
#                     best_params = params

#                 iteration += 1
#                 return

#             for value in param_values[depth]:
#                 grid_search_recursive(depth + 1, current_params + [(param_names[depth], value)])

#         grid_search_recursive(0, [])

#         tuning_results[model_name] = {"best_params": best_params, "best_score": best_score}
#         print(f"Best result for {model_name}: Params = {best_params}, Score = {best_score}\n")

#     return tuning_results


# def tune_hyperparameters1(models, param_grid, X, y, k=5):
#     """
#     Tune hyperparameters using grid search and k-fold cross-validation with minimal output.

#     Args:
#         models (dict): Dictionary with model names as keys and functions as values.
#         param_grid (dict): Dictionary of parameter grids for each model.
#         X (np.ndarray): Input features.
#         y (np.ndarray): Target values.
#         k (int): Number of folds for cross-validation.

#     Returns:
#         dict: Best hyperparameters and scores for each model.
#     """
#     tuning_results = {}

#     for model_name, model_fn in models.items():
#         print(f"\nCross-validating and tuning {model_name}...")
        
#         model_params = param_grid[model_name]
#         best_score = float('inf')
#         best_params = None
#         iteration = 0
#         param_combinations = 1

#         # Calculate the total number of parameter combinations and debug print
#         if model_params:
#             param_combinations = np.prod([len(values) for values in model_params.values()])
#         print(f"Total parameter combinations for {model_name}: {param_combinations}")

#         try:
#             # Recursive function for grid search
#             def grid_search_recursive(depth, current_params):
#                 nonlocal best_score, best_params, iteration
#                 param_names = list(model_params.keys())
#                 param_values = list(model_params.values())
                
#                 if depth == len(param_names):
#                     # Convert list of parameters to dictionary
#                     params = dict(current_params)
                    
#                     # Cross-validate with the current parameter combination
#                     cv_score = cross_validate(model_fn, X, y, k=k, **params)
                    
#                     # Print the current iteration number and parameters
#                     print(f"Iteration {iteration}: Params: {params}, Cross-validated loss: {cv_score}")
                    
#                     # Update best score and parameters if current score is better
#                     if cv_score < best_score:
#                         best_score = cv_score
#                         best_params = params
#                     iteration += 1
#                     return

#                 # Iterate over the values for the current hyperparameter
#                 for value in param_values[depth]:
#                     grid_search_recursive(depth + 1, current_params + [(param_names[depth], value)])

#             # Start grid search
#             grid_search_recursive(0, [])

#             # Store and print the best result for the current model
#             tuning_results[model_name] = {"best_params": best_params, "best_score": best_score}
#             print(f"Best result for {model_name}: Params = {best_params}, Score = {best_score}\n")

#         except Exception as e:
#             print(f"Error encountered in model {model_name}: {e}")
#             continue  # Skip to the next model if an error occurs

#     return tuning_results

# def tune_hyperparameters2(models, param_grid, X, y, k=5):
#     tuning_results = {}

#     for model_name, model_fn in models.items():
#         print(f"\nCross-validating and tuning {model_name}...")

#         model_params = param_grid[model_name]
#         best_score = float('inf')
#         best_params = None
#         iteration = 0

#         def grid_search_recursive(depth, current_params):
#             nonlocal best_score, best_params, iteration
#             param_names = list(model_params.keys())
#             param_values = list(model_params.values())

#             if depth == len(param_names):
#                 params = dict(current_params)
#                 cv_score = cross_validate(model_fn, X, y, k=k, **params)
#                 print(f"Iteration {iteration}: Params: {params}, Cross-validated loss: {cv_score}")

#                 if cv_score < best_score:
#                     best_score = cv_score
#                     best_params = params

#                 iteration += 1
#                 return

#             for value in param_values[depth]:
#                 grid_search_recursive(depth + 1, current_params + [(param_names[depth], value)])

#         grid_search_recursive(0, [])

#         tuning_results[model_name] = {"best_params": best_params, "best_score": best_score}
#         print(f"Best result for {model_name}: Params = {best_params}, Score = {best_score}\n")

#     return tuning_results