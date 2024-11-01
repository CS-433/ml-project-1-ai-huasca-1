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
    # Generate an array of indices and shuffle them to randomize the data folds
    indices = np.arange(len(y))
    np.random.shuffle(indices)

    # Define the size of each fold for splitting the data into k folds
    fold_size = len(y) // k
    # Create a list of k folds, each containing approximately equal-sized, randomized data indices
    folds = [indices[i * fold_size : (i + 1) * fold_size] for i in range(k)]

    # Add any remaining data points to the last fold if the data isn't perfectly divisible by k
    if len(y) % k != 0:
        folds[-1] = np.concatenate((folds[-1], indices[k * fold_size :]))

    # Initialize a list to store the loss for each fold
    losses = []

    # Iterate through each fold, using it as the validation set while the remaining k-1 folds are the training set
    for i in range(k):
        # Select indices for the validation set
        val_indices = folds[i]

        # Concatenate the remaining folds to form the training set
        train_indices = np.concatenate([folds[j] for j in range(k) if j != i])

        # Extract training and validation data based on the indices
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]

        # Call the model function; check if it requires initial weights and pass them if so
        if "initial_w" in model_fn.__code__.co_varnames:
            w, loss = model_fn(y_train, X_train, initial_w=initial_weights, **kwargs)
        else:
            w, loss = model_fn(y_train, X_train, **kwargs)

        # Compute the loss on the validation set using the trained weights
        val_loss = compute_loss(
            y_val, X_val, w
        )  # Assuming compute_loss is MSE or another loss function
        losses.append(val_loss)  # Store the validation loss for this fold

    # Calculate and return the average validation loss across all k folds
    return np.mean(losses)


def tune_hyperparameters(models, param_grid, X, y, initial_weights, k=5):
    """
    Tune hyperparameters for multiple models using grid search combined with k-fold cross-validation.

    This function evaluates different hyperparameter combinations for each model in `models` by performing
    k-fold cross-validation on all parameter combinations in `param_grid`. For each model, the best-performing
    hyperparameters (yielding the lowest cross-validated loss) are selected.

    Args:
        models (dict): Dictionary where keys are model names (str) and values are the corresponding model functions.
        param_grid (dict): Dictionary of parameter grids, where keys are model names and values are dictionaries of
                           parameter names and their possible values as lists.
        X (np.ndarray): Input features array of shape (N, D), where N is the number of samples and D is the number of features.
        y (np.ndarray): Target values array of shape (N,).
        initial_weights (np.ndarray): Initial weights array of shape (D,), used for models that require initialization.
        k (int, optional): Number of folds for cross-validation; defaults to 5.

    Returns:
        dict: A dictionary containing the best hyperparameters and corresponding scores for each model,
              structured as `{model_name: {"best_params": dict, "best_score": float}}`.

    Notes:
        - The `cross_validate` function is used to compute the cross-validated loss for each parameter combination.
        - This function supports recursive grid search over multiple parameters and values.
        - Initial weights are applied uniformly across models; models not requiring initial weights ignore this input.
        - The tuning process prints the cross-validated score for each parameter combination, tracking progress.

    """
    tuning_results = (
        {}
    )  # Dictionary to store the best hyperparameters and scores for each model

    # Iterate over each model in the dictionary
    for model_name, model_fn in models.items():
        print(f"\nTuning {model_name}...")  # Log the model currently being tuned

        model_params = param_grid[
            model_name
        ]  # Get the parameter grid specific to the model
        best_score = float(
            "inf"
        )  # Initialize best score to a very high value (lower scores are better)
        best_params = None  # Variable to store the best parameter set for the model

        # Define recursive function to search over all parameter combinations
        def recursive_grid_search(depth, current_params):
            nonlocal best_score, best_params
            param_names = list(model_params.keys())  # List of parameter names
            param_values = list(
                model_params.values()
            )  # Corresponding list of values for each parameter

            # If all parameters have been assigned a value, evaluate the current combination
            if depth == len(param_names):
                params = dict(
                    current_params
                )  # Convert list of tuples to dictionary format

                # Perform cross-validation with the current parameter combination
                cv_score = cross_validate(
                    model_fn, X=X, y=y, initial_weights=initial_weights, k=k, **params
                )
                print(f"Params: {params}, Cross-validated loss: {cv_score}")

                # Update best score and parameters if the current score is lower than previous best
                if cv_score < best_score:
                    best_score = cv_score
                    best_params = params  # Save the parameters that produced this score

                return

            # Recur over all values for the current parameter
            for value in param_values[depth]:
                # Pass along the current parameter combination with the new value added
                recursive_grid_search(
                    depth + 1, current_params + [(param_names[depth], value)]
                )

        # Start the recursive grid search with an empty parameter list
        recursive_grid_search(0, [])

        # Store the best parameters and score found for the model
        tuning_results[model_name] = {
            "best_params": best_params,
            "best_score": best_score,
        }
        print(
            f"Best for {model_name}: Params = {best_params}, Score = {best_score}\n"
        )  # Log final best result for the model

    return tuning_results  # Return the dictionary containing best hyperparameters and scores for each model
