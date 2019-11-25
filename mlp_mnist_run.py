import numpy as np, gzip, pickle
from mlp_io import *
from mlp_train import *
from mlp_predict import *

#====================
# Specify parameters
#====================

# Input
data_path = "data/mnist.pkl.gz"

# Learning rate and stopping
learning_rate_initial = 4 # 8 with no scaling
max_steps_down = 9
learning_rate_min = learning_rate_initial / 2**max_steps_down
max_epochs = 100

# Learnable hyperparameters
n_hidden_vals = [[100]] # Architecture of hidden layers
batch_size_vals = [40]
regulariser_vals = [0.01] # 0.035 with no scaling, 0.01 with

# Feature trimming and scaling
min_var_samples = 1000

# Decision switches
train_10bit = True
train_4bit = False
scale_features = True
verbose = True
profile = False
report_test = False

# TODO: Implement profiling
# TODO: Implement momentum optimisation

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

#=============================
# Drop uninformative features
#=============================

f_in = inputs_train.shape[1]
print("Dropping invariant features...", end="")
# Combine datasets
cut = np.cumsum([len(inputs_train), len(inputs_val)])
inputs_all = np.vstack([inputs_train, inputs_val, inputs_test])
# Discard invariant features (e.g. image padding)
inputs_variant = trim_features(inputs_all, min_var_samples)
# Re-separate datasets
inputs_train = inputs_variant[:cut[0],:]
inputs_val = inputs_variant[cut[0]:cut[1],:]
inputs_test = inputs_variant[cut[1]:,:]
print("done.")
f_out = inputs_train.shape[1]
print("{0} features ({1}%) discarded. {2} remaining.".format(f_in-f_out,
    round((f_in-f_out)/f_in * 100, 2), f_out))

#=====================================
# Apply feature scaling (all at once)
#=====================================

if scale_features:
    print("Scaling features...", end = "")
    inputs_train = feature_scale(inputs_train)
    inputs_val = feature_scale(inputs_val)
    inputs_test = feature_scale(inputs_test)
    print("done.")

#===================================
# Train and evaluate 10-bit network
#===================================

if train_10bit:
    print("\nTraining 10-bit network:\n")
    nn_10bit = train_hyperparameters(inputs_train, dec2bin10(labels_train),
            inputs_val, dec2bin10(labels_val), learning_rate_initial,
            learning_rate_min, max_epochs, n_hidden_vals,
            batch_size_vals, regulariser_vals, verbose)
    outputs_10bit = {"train": output(inputs_train, nn_10bit["weights"]),
            "val": output(inputs_val, nn_10bit["weights"]),
            "test": output(inputs_test, nn_10bit["weights"])}
    predict_10bit = {"train": bin102dec(outputs_10bit["train"]),
            "val": bin102dec(outputs_10bit["val"]),
            "test": bin102dec(outputs_10bit["test"])}
    error_10bit = {"test": get_pred_err(predict_10bit["test"], labels_test),
            "val": get_pred_err(predict_10bit["val"], labels_val),
            "train": get_pred_err(predict_10bit["train"], labels_train)}
    print("Training error:", round(error_10bit["train"]*100, 2), "%")
    print("Validation error:", round(error_10bit["val"]*100, 2), "%")
    if report_test: print("** Test error:", round(error_10bit["test"]*100, 2), "% **")

#===================================
# Train and evaluate 4-bit network
#===================================

if train_4bit:
    print("\nTraining 4-bit network:\n")
    nn_4bit = train_hyperparameters(inputs_train, dec2bin10(labels_train),
            inputs_val, dec2bin4(labels_val), learning_rate_initial,
            learning_rate_min, max_epochs, n_hidden_vals,
            batch_size_vals, regulariser_vals, verbose)
    outputs_4bit = {"train": output(inputs_train, nn_4bit["weights"]),
            "val": output(inputs_val, nn_4bit["weights"]),
            "test": output(inputs_test, nn_4bit["weights"])}
    predict_4bit = {"train": bin42dec(outputs_4bit["train"]),
            "val": bin42dec(outputs_4bit["val"]),
            "test": bin42dec(outputs_4bit["test"])}
    error_4bit = {"test": get_pred_err(predict_4bit["test"], labels_test),
            "val": get_pred_err(predict_4bit["val"], labels_val),
            "train": get_pred_err(predict_4bit["train"], labels_train)}
    print("Training error:", round(error_4bit["train"]*100, 2), "%")
    print("Validation error:", round(error_4bit["val"]*100, 2), "%")
    if report_test: print("** Test error:", round(error_4bit["test"]*100, 2), "% **")
