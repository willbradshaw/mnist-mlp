import numpy as np, torch, gzip, pickle
from mlp_network import *
from mlp_train import *
from mlp_io import *

#====================
# Specify parameters
#====================

# Decision switches
scale_features = True
verbose = True
profile = True
report_test = False
scale_by_range = True # Else by standard deviation

# Input
data_path = "data/mnist.pkl.gz"

# Learning rate and stopping
learning_rate_initial = 8.0
max_steps_down = 9 if scale_features else 9
learning_rate_min = learning_rate_initial / 2**max_steps_down
max_epochs = 100

# Learnable hyperparameters
# Most recent best: 360, 0.08, [327], 0.3 (TrE 0.18%, VE 2.03%, TeE 1.97%)
batch_size_vals = [10, 50] #[360]*1 if scale_features else [160]
regulariser_vals = [1e-5, 1e-4] #[0.08] if scale_features else [0.035]
momentum = [0.4] if scale_features else [0,0.1,0.3,0.5]
n_hidden_vals = [[100], [200]] #[[int(654/2)]] # Architecture of hidden layers

# Feature trimming and scaling
min_var_samples = 10 if ((not scale_features) or scale_by_range) else 100

#=============
# Import data
#=============

print("\nImporting data...", end="")
f = gzip.open(data_path, "rb")
try:
    data_train, data_val, data_test = pickle.load(f, encoding="latin1")
finally:
    f.close()
inputs_train, labels_train = [torch.tensor(d) for d in data_train]
inputs_val, labels_val = [torch.tensor(d) for d in data_val]
inputs_test, labels_test = [torch.tensor(d) for d in data_test]
print("done.")

#=============================
# Drop uninformative features
#=============================

f_in = inputs_train.shape[1]
print("Minimum # variant samples to retain feature:", min_var_samples)
print("Dropping invariant features...", end="")
# Combine datasets
cut = np.cumsum([len(inputs_train), len(inputs_val)])
inputs_all = torch.cat([inputs_train, inputs_val, inputs_test], 0)
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
    inputs_all = torch.cat([inputs_train, inputs_val, inputs_test], 0)
    # Scale features
    inputs_scaled = feature_scale(inputs_all)
    # Re-separate datasets
    inputs_train = inputs_scaled[:cut[0],:]
    inputs_val = inputs_scaled[cut[0]:cut[1],:]
    inputs_test = inputs_scaled[cut[1]:,:]
    print("done.")

#===============
# Train network
#===============

msg = "\nPerforming SGD with learning rate scheduling."
print(msg)
print("Initial learning rate:", learning_rate_initial)
print("Minimium learning rate:", learning_rate_min)
nn = train_hyperparameters(inputs_train, dec2bin10(labels_train),
        inputs_val, dec2bin10(labels_val), learning_rate_initial,
        learning_rate_min, max_epochs, n_hidden_vals,
        batch_size_vals, regulariser_vals, momentum, verbose, profile)

#==============================
# Evaluate network performance
#==============================

inputs = {"train": inputs_train, "val": inputs_val, "test": inputs_test}
labels = {"train": labels_train, "val": labels_val, "test": labels_test}
outputs = dict([(k, nn["network"].forward(inputs[k])) for k in inputs.keys()])
predictions = dict([(k, bin102dec(outputs[k])) for k in outputs.keys()])
errors = dict([(k, get_pred_err(predictions[k], labels[k])) for k in labels.keys()])
print("Training error:", round(errors["train"]*100,2), "%")
print("Validation error:", round(errors["val"]*100,2), "%")
if report_test: print("** Test error:", round(errors["test"]*100,2), "% **")