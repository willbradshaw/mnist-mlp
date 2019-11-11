import pickle, gzip, numpy as np
# TODO: Regularise everything
# TODO: Import data from internet

# Import data
# ...

# Convert labels
def dec2bin10(labels):
    """Convert decimal label vectors into 10-column binary matrices."""
    labels_out = np.zeros([len(labels),10])
    labels_out[np.arange(len(labels)), labels] = 1
    return labels_out

def dec2bin4(labels):
    """Convert decimal label vectors into 4-column binary matrices."""
    return np.array([list(np.binary_repr(x, 4)) \
            for x in labels]).astype("int64")

def bin42dec(matrix):
    """Convert 4-column binary matrix into a decimal array."""
    return (np.round(matrix).astype(int) * 2**np.arange(4)[::-1]).sum(axis=1)

def bin102dec(matrix):
    """Convert 10-column binary matrix into a decimal array."""
    return np.argmax(matrix, 1)
