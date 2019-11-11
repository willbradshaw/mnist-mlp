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
cost_threshold_rel = 0.0001
min_epochs = 0
max_epochs = 30
train_4bit = False
verbose = True

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

#=====================================
# Apply feature scaling (all at once)
#=====================================

cut = np.cumsum([len(inputs_train), len(inputs_val)])

inputs_all = np.vstack([inputs_train, inputs_val, inputs_test])
inputs_scale = feature_scale(inputs_all)

inputs_train = inputs_scale[:cut[0],:]
inputs_val = inputs_scale[cut[0]:cut[1],:]
inputs_test = inputs_scale[cut[1]:,:]

#===================================
# Train and evaluate 10-bit network
#===================================

print("Training 10-bit network:", "\n")
nn_10bit = train_hyperparameters(inputs_train, dec2bin10(labels_train),
        inputs_val, dec2bin10(labels_val), n_hidden_vals, learning_rate_vals,
        batch_size_vals, regulariser_vals, cost_threshold_rel, min_epochs,
        max_epochs, verbose)
outputs_10bit = {"train": output(inputs_train, nn_10bit["weights"]),
        "val": output(inputs_val, nn_10bit["weights"]),
        "test": output(inputs_test, nn_10bit["weights"])}
predict_10bit = {"train": bin102dec(outputs_10bit["train"]),
        "val": bin102dec(outputs_10bit["val"]),
        "test": bin102dec(outputs_10bit["test"])}
error_10bit = {"test": get_pred_err(predict_10bit["test"], labels_test),
        "val": get_pred_err(predict_10bit["val"], labels_val),
        "train": get_pred_err(predict_10bit["train"], labels_train)}
print("Training error:", error_10bit["train"])
print("Validation error:", error_10bit["val"])
print("** Test error:", error_10bit["test"], "**")

#===================================
# Train and evaluate 4-bit network
#===================================

if train_4bit:
    print("Training 4-bit network:", "\n")
    nn_4bit = train_hyperparameters(inputs_train, dec2bin10(labels_train),
            inputs_val, dec2bin10(labels_val), n_hidden_vals, learning_rate_vals,
            batch_size_vals, regulariser_vals, cost_threshold_rel, min_epochs,
            max_epochs, verbose)
    outputs_4bit = {"train": output(inputs_train, nn_4bit["weights"]),
            "val": output(inputs_val, nn_4bit["weights"]),
            "test": output(inputs_test, nn_4bit["weights"])}
    predict_4bit = {"train": bin42dec(outputs_4bit["train"]),
            "val": bin42dec(outputs_4bit["val"]),
            "test": bin42dec(outputs_4bit["test"])}
    error_4bit = {"test": get_pred_err(predict_4bit["test"], labels_test),
            "val": get_pred_err(predict_4bit["val"], labels_val),
            "train": get_pred_err(predict_4bit["train"], labels_train)}
    print("Training error:", error_4bit["train"])
    print("Validation error:", error_4bit["val"])
    print("** Test error:", error_4bit["test"], "**")
