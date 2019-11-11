import numpy as np, copy, math, itertools
# TODO: Regularise everything
# TODO: Import data from internet

def sigmoid(matrix):
    """Apply the sigmoid function to a matrix."""
    return 1/(1+np.exp(-matrix))

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
        regulariser):
    """Compute regularised gradients for MLP weights."""
    # TODO: Check correspondence between deltas and activations
    sample_size = len(inputs)
    activations_list = [inputs] + activations_list
    grads_list = [(np.matmul(deltas_list[n].T, add_bias(activations_list[n]))+\
            regulariser * zero_bias(weights_list[n]))/sample_size \
            for n in range(len(weights_list))]
    return grads_list

def sigmoid_cost(outputs, labels, weights_list, regulariser):
    """Compute the cost function of a sigmoid MLP net."""
    inner_cost = -np.sum(labels*np.log(outputs) + (1-labels)*np.log(1-outputs))
    reg_cost = sum([np.sum(weights_list[n]**2) \
            for n in range(len(weights_list))]) * regulariser / 2
    return((inner_cost + reg_cost)/len(labels))

def compute_gradients_numeric_single(inputs, labels, weights_list, n,
        epsilon=1e-4):
    """Compute numerical gradients for a single weight matrix."""
    weights = weights_list[n]
    size = weights.size
    grads = np.zeros(weights.shape)
    indices = np.array(np.unravel_index(range(size), weights.shape))
    for i in range(size):
        add = np.zeros(weights_list[n].shape)
        add[indices[0,i],indices[1,i]] = epsilon
        weights_list_up = copy.deepcopy(weights_list)
        weights_list_down = copy.deepcopy(weights_list)
        weights_list_up[n] += add
        weights_list_down[n] -= add
        outputs_up = forward_propagation(inputs, weights_list_up)[-1]
        outputs_down = forward_propagation(inputs, weights_list_down)[-1]
        cost_up = sigmoid_cost(outputs_up, labels, weights_list_up, 0)
        cost_down = sigmoid_cost(outputs_down, labels, weights_list_down, 0)
        grads[indices[0,i],indices[1,i]] = (cost_up - cost_down)/(2*epsilon)
    return(grads)

def compute_gradients_numeric(inputs, labels, weights_list, epsilon=1e-4):
    """Compute unregularised gradients using numerical estimation."""
    grads_list = [None] * len(weights_list)
    for n in range(len(weights_list)):
        grads_list[n] = compute_gradients_numeric_single(inputs, labels,
                weights_list, n, epsilon)
    return(grads_list)

def update_weights(weights_list, grads_list, learning_rate):
    """Update weights from activations and deltas."""
    return [weights_list[n] - learning_rate * grads_list[n] \
            for n in range(len(weights_list))]

def get_starting_weights(inputs, labels, n_hidden):
    """Randomly initialise starting weights based on stated architecture."""
    architecture = [inputs.shape[1]] + n_hidden + [labels.shape[1]]
    weights_list = [np.random.randn(architecture[n+1], architecture[n] + 1) \
            for n in range(len(architecture))[:-1]]
    # TODO: Reduce width of distribution?
    return(weights_list)

def descend(inputs, labels, weights_list, learning_rate, regulariser):
    """Perform one iteration of gradient descent."""
    activations_list = forward_propagation(inputs, weights_list)
    deltas_list = backpropagation(activations_list, weights_list, labels)
    grads_list = compute_gradients(weights_list, inputs, activations_list,
            deltas_list, regulariser)
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
                weights_list, learning_rate, regulariser)
        batch += batch_size
        batch = batch[batch < sample_size]
        n += 1
    return([weights_list, costs])

def gradient_descent(inputs, labels, n_hidden, learning_rate, batch_size,
        regulariser, cost_threshold_rel, min_epochs, max_epochs, verbose):
    """Perform gradient descent to convergence."""
    # TODO: Better way to decide when to stop?
    # Initialise
    batch = np.arange(batch_size)
    weights_list = get_starting_weights(inputs[batch], labels[batch], n_hidden)
    n = 0
    converged = False
    # Run first epoch
    if verbose: print("   ", "   ", "epoch:", n)
    [weights_list, costs] = descend_epoch(inputs, labels, weights_list,
            learning_rate, batch_size, regulariser)
    # Run algorithm until convergence
    while converged == False:
        n += 1
        if verbose: print("   ", "   ", "epoch:", n)
        [weights_list, costs_new] = descend_epoch(inputs, labels, weights_list,
                learning_rate, batch_size, regulariser)
        costs = np.vstack([costs,costs_new])
        cost_ratio = costs[-1,-1]/costs[-2,-1]
        if n >= max_epochs or ((1 - cost_ratio) <= cost_threshold_rel and n >= min_epochs):
            converged = True
    if verbose: print("   ", "   ", "Final training cost:", costs[-1,-1])
    return({"weights":weights_list, "costs":costs})

def train_hyperparameters(inputs_train, labels_train, inputs_val, labels_val,
        n_hidden_vals, learning_rate_vals, batch_size_vals, regulariser_vals,
        cost_threshold_rel, min_epochs, max_epochs, verbose):
    """Train MLP using various hyperparameter values and pick the best ones
    using the validation dataset."""
    combs = itertools.product(learning_rate_vals, batch_size_vals,
            regulariser_vals, n_hidden_vals)
    cost = math.inf
    n = 0
    for learning_rate,batch_size,regulariser,n_hidden in combs:
        print("   ", "Hyperparameters:",learning_rate,batch_size,regulariser,n_hidden)
        nn = gradient_descent(inputs_train, labels_train, n_hidden,
                learning_rate, batch_size, regulariser, cost_threshold_rel,
                min_epochs, max_epochs, verbose)
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
