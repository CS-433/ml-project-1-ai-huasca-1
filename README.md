AI Huasca Team: Henry Schultz 342151, Louis Tschanz 315774, Majandra Garcia 347470
henry.schultz@epfl.ch   louis.tschanz@epfl.ch   majandra.garcia@epfl.ch

# Machine Learning Project

This project implements a comprehensive machine learning pipeline to preprocess data, train models, and evaluate their performance through cross-validation and hyperparameter tuning. It is structured with modular components for flexibility and reuse, making it suitable for various machine learning tasks.

## Project Structure

The project is organized as follows:

- **helpers_perso/**  
  Contains custom helper functions that facilitate specific tasks in the pipeline.
  
  - **helpers_implementations.py**: Provides utility functions that support core model implementations, including commonly used mathematical and data manipulation functions.
  - **helpers_nan_imputation.py**: Focuses on handling missing values, offering methods to impute NaN values based on various strategies such as mean, median, or mode.

- **preprocessing/**  
  This folder houses modules for data preprocessing.

  - **class_balancing.py**: Contains the method to handle the imbalance of the dataset.
  - **nan_imputation.py**: Provides functions to impute missing values, allowing the user to specify imputation strategies to handle incomplete data effectively.
  - **binary_encoding.py**: Handles categorical data encoding, transforming categorical variables into a format that can be provided to ML algorithms.
  - **remove_highly_correlated_features.py**: Identifies and removes features that are highly correlated, reducing redundancy and potential model overfitting. (not used in the final pipeline)
  - **standardization.py**: Standardizes quantitative features.
  - **preprocessing.ipynb**: A Jupyter Notebook used to develop and tune the preprocessing steps, providing an interactive environment for testing and visualization.

- **implementations.py**  
  This file contains the main implementations of the machine learning algorithms used in this project.

- **run.py**  
  The main script for executing the project pipeline. It integrates preprocessing, model training, classification and submission file writing.

- **crossvalidation.py**  
  Implements cross-validation functions for model assessment.

- **Cross_Validation_and_Tuning.ipynb**  
  A Jupyter Notebook dedicated to exploring various hyperparameter tuning techniques and cross-validation results. It provides an interactive environment to visualize and analyze model performance across different hyperparameter configurations.

- **predict_labels.py**  
  Contains functions specifically for generating predictions on new data, translating model outputs into labels that can be interpreted in the context of the application.

- **prediction_score.py**  
  Evaluates model predictions by calculating metrics such as accuracy, precision, recall, or F1 score. This script helps assess model effectiveness and provides insights into areas for improvement.

## Getting Started

### Prerequisites
Ensure the following packages are installed:
- `numpy`
- `matplotlib`
  
Install any missing dependencies via `pip install`.

### Running the Project

1. **Data Preprocessing**: Use `preprocessing.ipynb` to experiment on preprocessing of your dataset or run individual scripts within the `preprocessing/` folder.
2. **Model Training and Evaluation**: Execute `run.py` for writing a submission file.
3. **Hyperparameter Tuning**: Use `Cross_Validation_and_Tuning.ipynb` to explore and fine-tune model parameters for optimal performance.

