import numpy as np, copy, math, itertools
# TODO: Regularise everything
# TODO: Import data from internet

def sigmoid(matrix):
    """Apply the sigmoid function to a matrix."""
    # Correct for overflow
    #max_val = np.log(np.finfo(matrix.dtype).max)-1e-5
    #matrix_clipped = np.clip(matrix, None, max_val)
    return 1/(1+np.exp(-np.nan_to_num(matrix)))

def add_bias(matrix):
    """Add a bias-unit column to an input or activation matrix."""
    return np.hstack([np.ones([matrix.shape[0], 1]), matrix])
    # TODO: Check efficiency of hstack vs alternatives

def drop_bias(matrix):
    """Remove the bias-unit column from a weight or delta matrix."""
    return matrix[:,1:]

def zero_bias(matrix):
    """Convert the bias-unit column from a weight or delt matrix to zeros."""
    matrix[:,0] = 0
    return(matrix)

def linear_activate(inputs, weights):
    """Linearly combine inputs and weights."""
    return(np.matmul(add_bias(inputs), weights.T))

def sigmoid_activate(inputs, weights):
    """Compute the activation of a sigmoid neuron."""
    return sigmoid(linear_activate(inputs, weights))

def forward_propagation(inputs, weights_list):
    """Update activation matrices from inputs and weights."""
    activations_list = [inputs] + [None] * len(weights_list)
    for n in range(len(weights_list)):
        activations = sigmoid_activate(activations_list[n],
                weights_list[n])
        activations_list[n+1] = activations
    return(activations_list[1:])

def backpropagation(activations_list, weights_list, labels):
    """Infer delta matrices from activations, weights and labels."""
    assert len(activations_list) == len(weights_list)
    N = len(activations_list)
    deltas_list = [None] * N
    deltas_list[-1] = activations_list[-1] - labels
    for n in range(N)[:-1]: # TODO: Check range
        m = -(n+1)
        comp1 = drop_bias(np.matmul(deltas_list[m], weights_list[m]))
        comp2 = activations_list[m-1] * (1 - activations_list[m-1])
        deltas_list[m-1] = comp1 * comp2
    return(deltas_list)

def compute_gradients(weights_list, inputs, activations_list, deltas_list,
        regulariser, total_len):
    """Compute regularised gradients for MLP weights."""
    # TODO: Check correspondence between deltas and activations
    sample_size = len(inputs)
    activations_list = [inputs] + activations_list
    grads_list = [None] * len(weights_list)
    for n in range(len(weights_list)):
        grad_unreg = np.matmul(deltas_list[n].T, add_bias(activations_list[n]))\
                /sample_size
        grad_reg = (regulariser/total_len) * zero_bias(weights_list[n])
        grads_list[n] = grad_unreg + grad_reg
    return grads_list

def sigmoid_cost(outputs, labels, weights_list, regulariser):
    """Compute the cost function of a sigmoid MLP net."""
    inner_cost = -np.sum(np.nan_to_num(labels*np.log(outputs) + \
            (1-labels)*np.log(1-outputs)))
    reg_cost = sum([np.sum(zero_bias(weights_list[n])**2) \
            for n in range(len(weights_list))]) * regulariser / 2
    return((inner_cost + reg_cost)/len(labels))

def update_weights(weights_list, grads_list, learning_rate):
    """Update weights from activations and deltas."""
    return [weights_list[n] - learning_rate * grads_list[n] \
            for n in range(len(weights_list))]

def get_starting_weights(inputs, labels, n_hidden):
    """Randomly initialise starting weights based on stated architecture."""
    architecture = [inputs.shape[1]] + n_hidden + [labels.shape[1]]
    weights_list = [None] * (len(architecture)-1)
    for n in range(len(architecture))[:-1]:
        weights_unscaled = np.random.randn(architecture[n+1], architecture[n]+1)
        weights_scaled = weights_unscaled / np.sqrt(architecture[n])
        weights_list[n] = weights_scaled
    return(weights_list)

def descend(inputs, labels, weights_list, learning_rate, regulariser,
        total_len):
    """Perform one iteration of gradient descent."""
    activations_list = forward_propagation(inputs, weights_list)
    deltas_list = backpropagation(activations_list, weights_list, labels)
    grads_list = compute_gradients(weights_list, inputs, activations_list,
            deltas_list, regulariser, total_len)
    weights_list = update_weights(weights_list, grads_list, learning_rate)
    cost = sigmoid_cost(activations_list[-1], labels, weights_list, regulariser)
    return([weights_list, cost])

def descend_epoch(inputs, labels, weights_list, learning_rate, batch_size,
        regulariser):
    """Perform one epoch of gradient descent."""
    sample_size = len(inputs)
    order = np.random.choice(np.arange(len(inputs)), len(inputs), False)
    inputs = inputs[order]
    labels = labels[order]
    n = 0
    batch = np.arange(batch_size)
    costs = np.zeros(math.ceil(sample_size/batch_size))
    while len(batch) > 0:
        [weights_list,costs[n]] = descend(inputs[batch], labels[batch],
                weights_list, learning_rate, regulariser, len(labels))
        batch += batch_size
        batch = batch[batch < sample_size]
        n += 1
    return([weights_list, costs])

def gradient_descent(inputs, labels, n_hidden, learning_rate, batch_size,
        regulariser, n_epochs, verbose):
    """Perform n complete epochs of stochastic gradient descent and return
    the best."""
    # Initialise
    batch = np.arange(batch_size)
    weights_list = [None] * n_epochs
    n_batches = np.ceil(len(inputs)/batch_size).astype(int)
    costs = np.zeros([n_epochs,n_batches])
    weights_init = get_starting_weights(inputs[batch], labels[batch], n_hidden)
    # Run first epoch
    if verbose: print("   ", "   ", "Epoch 1...", end="")
    [weights_list[0], costs[0]] = descend_epoch(inputs, labels, weights_init,
            learning_rate, batch_size, regulariser)
    if verbose: print("done.")
    # Run remaining epochs
    for n in range(1, n_epochs):
        if verbose: print("   ", "   ", "Epoch {}...".format(n+1), end="")
        [weights_list[n], costs[n,:]] = descend_epoch(inputs, labels,
                weights_list[n-1], learning_rate, batch_size, regulariser)
        if verbose: print("done.")
    # Determine best output and return
    best_epoch = np.argmin(costs[:,-1])
    print("   ", "   ", "Best epoch:", best_epoch+1)
    return({"weights":weights_list[best_epoch], "costs":costs[best_epoch]})

def train_hyperparameters(inputs_train, labels_train, inputs_val, labels_val,
        n_hidden_vals, learning_rate_vals, batch_size_vals, regulariser_vals,
        n_epochs, verbose):
    """Train MLP using various hyperparameter values and pick the best ones
    using the validation dataset."""
    combs = itertools.product(learning_rate_vals, batch_size_vals,
            regulariser_vals, n_hidden_vals)
    cost = math.inf
    n = 0
    for learning_rate,batch_size,regulariser,n_hidden in combs:
        print("   ", "Hyperparameters:",learning_rate,batch_size,regulariser,n_hidden)
        nn = gradient_descent(inputs_train, labels_train, n_hidden,
                learning_rate, batch_size, regulariser, n_epochs, verbose)
        output_val = forward_propagation(inputs_val, nn["weights"])[-1]
        cost_val = sigmoid_cost(output_val, labels_val, nn["weights"], 0)
        if cost_val < cost:
            out = {"learning_rate":learning_rate, "batch_size":batch_size,
                    "regulariser":regulariser, "weights":nn["weights"],
                    "trace":nn["costs"], "n_hidden":n_hidden}
            cost = cost_val
    print("Best hyperparameters:", out["learning_rate"],
            out["batch_size"], out["regulariser"], out["n_hidden"])
    if verbose: print("Best validation error:", cost)
    return out
