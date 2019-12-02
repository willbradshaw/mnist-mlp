import pickle, gzip, torch

# Convert labels
def dec2bin10(labels):
    """Convert decimal label vectors into 10-column binary matrices."""
    labels_out = torch.zeros([len(labels),10])
    labels_out[torch.arange(len(labels)), labels] = 1
    return labels_out

def bin102dec(matrix):
    """Convert 10-column binary matrix into a decimal array."""
    return torch.argmax(matrix, 1)

# Process inputs
def trim_features(inputs, min_samples = 10):
    """Discard invariant features."""
    variant = torch.sum(inputs != torch.median(inputs, 0)[0], 0) >= min_samples
    return inputs[:,variant]

def feature_scale(inputs, scale_by_range = False):
    """Discard invariant features and process the rest to have uniform means
    and variances."""
    # Normalise by mean
    inputs = inputs - torch.mean(inputs, 0)
    # Rescale by SD
    scale_function = torch.std
    if scale_by_range:
        scale_function = lambda x: torch.max(x) - torch.min(x)
    inputs = inputs/scale_function(inputs, 0)
    return(inputs)
    
# Get prediction error
def get_pred_err(predictions, labels):
    """Compute the prediction error frequency for a neural network."""
    return float(1 - (predictions == labels).double().mean())