import numpy as np, gzip, pickle
from mlp_io import *
from mlp_train import *

#====================
# Specify parameters
#====================

data_path = "data/mnist.pkl.gz"
regulariser = 0.3
learning_rate = 1
batch_size = 10
n_hidden = [15]
cost_threshold_rel = 0.01
epsilon = 1e-4

# TODO: Try different hyperparameter values

#=============
# Import data
#=============

f = gzip.open(data_path, "rb")
try:
    data_train, data_val, data_test = pickle.load(f, encoding="latin1")
finally:
    f.close()
inputs_train, labels_train = data_train
inputs_val, labels_val = data_val
inputs_test, labels_test = data_test

#===========================
# Perform gradient checking
#===========================

batch = np.arange(batch_size)
inputs, labels = inputs_train, dec2bin10(labels_train)
weights_list_num = get_starting_weights(inputs[batch], labels[batch], n_hidden)
activations_list = forward_propagation(inputs, weights_list_num)
deltas_list = backpropagation(activations_list, weights_list_num, labels)
grads_list_backprop = compute_gradients(weights_list_num, inputs, activations_list,
        deltas_list, 0)
grads_list_numeric = compute_gradients_numeric(inputs, labels, weights_list_num,
        epsilon)

#=========================
# Train 10-output network
#=========================

#nn_10bit = gradient_descent(inputs_train, dec2bin10(labels_train),
#        n_hidden, learning_rate, batch_size, regulariser, cost_threshold_rel)
#weights_10bit = nn_10bit["weights"]
#trace_10bit = nn_10bit["costs"]


