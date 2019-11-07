import pickle, gzip, numpy as np
# TODO: Regularise everything
# TODO: Import data from internet

# Import data
# ...

# Convert labels
def dec2bin10(labels):
    """Convert decimal label vectors into 10-column binary matrices."""
    # Doing it the dumb for-loop-y way for now
    # TODO: Make this smarter
    labels_out = np.zeros([len(labels),10])
    labels_out[np.arange(len(labels)), labels] = 1
    return labels_out

def dec2bin4(labels):
    """Convert decimal label vectors into 4-column binary matrices."""
    return np.array([list(np.binary_repr(labels, 4)) \
            for x in labels]).astype("int64")

