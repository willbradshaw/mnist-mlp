import numpy as np, gzip, pickle
from mlp_io import *
from mlp_train import *
from mlp_predict import *

#====================
# Specify parameters
#====================

data_path = "data/mnist.pkl.gz"
regulariser_vals = [0, 0.1, 0.3, 1, 3]
learning_rate_vals = [0.1, 0.3, 1, 3]
batch_size_vals = [10]
n_hidden_vals = [[100]]
cost_threshold_rel = 0.001
min_epochs = 10
max_epochs = 30

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

#batch = np.arange(batch_size)
#inputs, labels = inputs_train, dec2bin10(labels_train)
#weights_list_num = get_starting_weights(inputs[batch], labels[batch], n_hidden)
#activations_list = forward_propagation(inputs, weights_list_num)
#deltas_list = backpropagation(activations_list, weights_list_num, labels)
#grads_list_backprop = compute_gradients(weights_list_num, inputs, activations_list,
#        deltas_list, 0)
#grads_list_numeric = compute_gradients_numeric(inputs, labels, weights_list_num,
#        epsilon)

#=========================
# Train 10-output network
#=========================

print("Training 10-bit network:", "\n")
nn_10bit = train_hyperparameters(inputs_train, dec2bin10(labels_train),
        inputs_val, dec2bin10(labels_val), n_hidden_vals, learning_rate_vals,
        batch_size_vals, regulariser_vals, cost_threshold_rel, min_epochs, max_epochn)
print("\nTraining 4-bit network:")
nn_4bit = train_hyperparameters(inputs_train, dec2bin4(labels_train),
        inputs_val, dec2bin4(labels_val), n_hidden_vals, learning_rate_vals,
        batch_size_vals, regulariser_vals, cost_threshold_rel, min_epochs, max_epochs)

#=========================
# Evaluate performance
#=========================

# Outputs
outputs_10bit = {"train": output(inputs_train, nn_10bit["weights"]),
        "val": output(inputs_val, nn_10bit["weights"]),
        "test": output(inputs_test, nn_10bit["weights"])}
outputs_4bit = {"train": output(inputs_train, nn_4bit["weights"]),
        "val": output(inputs_val, nn_4bit["weights"]),
        "test": output(inputs_test, nn_4bit["weights"])}

# Predictions
predict_10bit = {"train": bin102dec(outputs_10bit["train"]),
        "val": bin102dec(outputs_10bit["val"]),
        "test": bin102dec(outputs_10bit["test"])}
predict_4bit = {"train": bin42dec(outputs_4bit["train"]),
        "val": bin42dec(outputs_4bit["val"]),
        "test": bin42dec(outputs_4bit["test"])}

# Performance (prediction accuracy/error)
performance_10bit = np.mean(predict_10bit["test"] == labels_test)
performance_4bit = np.mean(predict_4bit["test"] == labels_test)
