[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/MqChnODK)

preprocessing parameters : 
-percentage of Nan in a column to drop it (maybe not even needed so in this case put =1)
-how to encode Nan in integer columns : 0/categorie N+1/mode value
-how to encode Nan in float columns : 0/mean/mode

AI Huasca Team: Henry Schultz, Louis Tschanz, Majandra Garcia

# Cardiovascular Disease Prediction Using Machine Learning

## Project Description
This project applies machine learning algorithms to predict the likelihood of coronary heart disease (CVD) based on health and lifestyle features. Using a dataset from the Behavioral Risk Factor Surveillance System (BRFSS), we train and evaluate several models to classify individuals as at risk or not at risk for developing CVD.

## Project Structure
- `data/`: Contains the dataset files (e.g., `x_train.csv`, `y_train.csv`, `x_test.csv`).
- `helpers.py`: Helper functions for loading data, creating CSV submissions, and other utilities.
- `implementations.py`: Contains core functions for the machine learning algorithms implemented, such as `gradient_descent`, `logistic_regression`, and `ridge_regression`.
- `README.md`: Overview of the project and instructions for use.
- `latex-template.tex`: Template for the project report in LaTeX.
- `report.pdf`: Final report with project findings, methods, and conclusions.

## Installation and Requirements
- Python 3.x
- Required libraries are listed in `requirements.txt`. Install them using:
  ```bash
  pip install -r requirements.txt
