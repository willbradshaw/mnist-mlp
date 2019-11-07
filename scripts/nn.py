import numpy as np
# TODO: Regularise everything
# TODO: Import data from internet

def sigmoid(matrix):
    """Apply the sigmoid function to a matrix."""
    return 1/(1+np.exp(-matrix))

def add_bias(matrix):
    """Add a bias-unit column to an input or activation matrix."""
    return np.hstack([np.ones([matrix.shape[0], 1]), matrix])
    # TODO: Check efficiency of hstack vs alternatives
    
def drop_bias(matrix):
    """Remove the bias-unit column from a weight or delta matrix."""
    return matrix[:,1:]

def linear_activate(inputs, weights):
    """Linearly combine inputs and weights."""
    return(np.matmul(add_bias(inputs), weights.T))

def sigmoid_activate(inputs, weights):
    """Compute the activation of a sigmoid neuron."""
    sigmoid(linear_activate(inputs, weights))

def forward_propagation(inputs, weights_list):
    """Update activation matrices from inputs and weights."""
    activations_list = [inputs]
    for n in range(len(weights_list)):
        activations_list[n+1] = sigmoid_activate(activations_list[n],
                weights_list[n])
    return(activations_list[1:])

def backpropagation(activations_list, weights_list, labels):
    """Infer delta matrices from activations, weights and labels."""
    assert len(activations_list) == len(weights_list)
    N = len(activations_list)
    deltas_list = [activations_list[-1] - labels] # Final layer delta
    for n in range(N):
        deltas_list[n+1] = drop_bias(np.matmul(deltas_list[n], weights_list[n])) * \
                activations_list[N - n - 1] * (1 - activations_list[N - n - 1])
    assert length(deltas_list) == length(activations_list)
    return(deltas_list)

def update_weights(weights_list, activations_list, deltas_list):
    """Update weights from activations and deltas."""
    for n in range(len(weights_list)):
        gradient = 


def get_backprop_gradients


def backpropagation(activations, weights, labels):
    """Update gradient matrices from activations, weights and labels."""
    # Get deltas








