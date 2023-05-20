import numpy as np

batch = [[1.0, 2.0, 3.0, 4.5],
         [-0.5, 2.1, 9.4, 2.2],
         [12.2, 5.4, 123.0, -1.0]]
weights = [[0.2, 0.8, -0.5, 1],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]

layer1_outputs = np.dot(batch, np.array(weights).T) + biases

print(layer1_outputs)

weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]
biases2 = [-1, 2, -0.5]

layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

print(layer2_outputs)

