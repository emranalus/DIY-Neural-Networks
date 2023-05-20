# Dynamicly Scaleable Neural Network

# Network Model
inputs = [1, 2, 3, 2.5]
weigths = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]

# Iterative Calculation
layer_outputs = []

# For each neuron
for neuron_weigths, neuron_bias in zip(weigths, biases):
    # To use it inside and outside of the for loop.
    neuron_output = 0

    # For each input and weigth of the neuron.
    for n_input, weigth in zip(inputs, neuron_weigths):
        neuron_output += n_input * weigth
    
    # 
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)

print(layer_outputs)

