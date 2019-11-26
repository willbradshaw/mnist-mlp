import numpy as np, gzip, pickle
from mlp_io import *
from mlp_train import *
from mlp_predict import *

#====================
# Specify parameters
#====================

# Decision switches
scale_features = True
verbose = True
profile = True
report_test = False
scale_by_range = True

# Input
data_path = "data/mnist.pkl.gz"

# Learning rate and stopping
learning_rate_initial = 8.0 if scale_features else 8.0
max_steps_down = 9 if scale_features else 9
learning_rate_min = learning_rate_initial / 2**max_steps_down
max_epochs = 100

# Learnable hyperparameters
n_hidden_vals = [[100]] # Architecture of hidden layers
batch_size_vals = [40] if scale_features else [160]
regulariser_vals = [0.04] if scale_features else [0.035]
momentum = [0.1] if scale_features else [0,0.1,0.3,0.5]

# Feature trimming and scaling
min_var_samples = 10 if ((not scale_features) or scale_by_range) else 100

# TODO: Implement profiling
# TODO: Implement momentum optimisation

#=============
# Import data
#=============

print("\nImporting data...", end="")
f = gzip.open(data_path, "rb")
try:
    data_train, data_val, data_test = pickle.load(f, encoding="latin1")
finally:
    f.close()
inputs_train, labels_train = data_train
inputs_val, labels_val = data_val
inputs_test, labels_test = data_test
print("done.")

#=============================
# Drop uninformative features
#=============================

f_in = inputs_train.shape[1]
print("Minimum # variant samples to retain feature:", min_var_samples)
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
    ref = "range" if scale_by_range else "SD"
    print("Scaling features by {}...".format(ref), end = "")
    # Combine datasets
    cut = np.cumsum([len(inputs_train), len(inputs_val)])
    inputs_all = np.vstack([inputs_train, inputs_val, inputs_test])
    # Scale features
    inputs_scaled = feature_scale(inputs_all)
    # Re-separate datasets
    inputs_train = inputs_scaled[:cut[0],:]
    inputs_val = inputs_scaled[cut[0]:cut[1],:]
    inputs_test = inputs_scaled[cut[1]:,:]
    print("done.")

#===================================
# Train and evaluate 10-bit network
#===================================

msg = "\nPerforming SGD with learning rate scheduling."
print(msg)
print("Initial learning rate:", learning_rate_initial)
print("Minimium learning rate:", learning_rate_min)
nn_10bit = train_hyperparameters(inputs_train, dec2bin10(labels_train),
        inputs_val, dec2bin10(labels_val), learning_rate_initial,
        learning_rate_min, max_epochs, n_hidden_vals,
        batch_size_vals, regulariser_vals, momentum, verbose, profile)
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