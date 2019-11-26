import numpy as np
from mlp_train import *

# Predict classes from weights
def output(inputs, weights, biases):
    """Get the output of a trained MLP for a given set of inputs."""
    return forward_propagation(inputs, weights, biases)[-1]

def predict_single(output):
    """Convert MLP outputs into class predictions, assuming one
    output neuron per class."""
    return np.argmax(output, 1)

# Evaluate network predictions

def get_pred_err(predictions, labels):
    """Compute the prediction error frequency for a neural network."""
    return 1 - np.mean(predictions == labels)
