import os
from helpers import load_csv_data
import numpy as np
import matplotlib.pyplot as plt
import implementations as imp

# Loading the data
data_path = os.path.join(os.getcwd(), 'dataset')
print(data_path)
x_train, x_test, y_train, train_ids, test_ids = load_csv_data(data_path)



### Testing imprementation 1
print('Implementation 1: mean_squared_error_gd \n')

# Initial weights (can be zeros, random, or some heuristic value)
initial_w = np.zeros(x_train.shape[1])

# Parameters for gradient descent
max_iters = 1000
gamma = 0.1  # Learning rate

# Running gradient descent with MSE
w_final_gd, loss_final_gd = imp.mean_squared_error_gd(y_train, x_train, initial_w, max_iters, gamma)

print(f"Final weights: {w_final_gd}")
print(f"Final loss: {loss_final_gd}")



### Testing imprementation 2
print('Implementation 2: mean_squared_error_sgd \n')

batch_size = 32  # Size of mini-batch for stochastic updates

# Running stochastic gradient descent with MSE
w_final_sgd, losses_final_sgd = imp.mean_squared_error_sgd(y_train, x_train, initial_w, max_iters, gamma, batch_size)

print(f"Final weights: {w_final_sgd}")
print(f"Final loss: {losses_final_sgd[-1]}")



### Testing implementation 3
print('Implementation 3: least_squares \n')

# Calculate the least squares solution
w_optimal, mse = imp.least_squares(y_train, x_train)

print(f"Optimal weights: {w_optimal}")
print(f"Mean Squared Error: {mse}")
