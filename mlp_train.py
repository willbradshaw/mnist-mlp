import numpy as np, copy, math, itertools, cProfile, pstats

def sigmoid(matrix, maxval = 1e100):
    """Apply the sigmoid function to a matrix."""
    # Correct for overflow
    max_val = np.log(np.finfo(matrix.dtype).max)
    matrix_clipped = np.clip(matrix, -max_val, max_val)
    return 1/(1+np.exp(-matrix_clipped))

def add_bias(matrix):
    """Add a bias-unit column to an input or activation matrix."""
    return np.hstack([np.ones([matrix.shape[0], 1]), matrix])

def drop_bias(matrix):
    """Remove the bias-unit column from a weight or delta matrix."""
    return matrix[:,1:]

def zero_bias(matrix):
    """Convert the bias-unit column from a weight or delt matrix to zeros."""
    matrix[:,0] = 0
    return(matrix)

def linear_activate(inputs, weights, biases):
    """Linearly combine inputs and weights."""
    return(np.matmul(inputs, weights.T) + biases.T)

def sigmoid_activate(inputs, weights, biases):
    """Compute the activation of a sigmoid neuron."""
    return sigmoid(linear_activate(inputs, weights, biases))

def forward_propagation(inputs, weights_list, bias_list):
    """Update activation matrices from inputs and weights."""
    activations_list = [inputs] + [None] * len(weights_list)
    for n in range(len(weights_list)):
        activations = sigmoid_activate(activations_list[n],
                weights_list[n], bias_list[n])
        activations_list[n+1] = activations
    return(activations_list[1:])

def backpropagation(activations_list, weights_list, labels):
    """Infer delta matrices from activations, weights and labels."""
    assert len(activations_list) == len(weights_list)
    N = len(activations_list)
    deltas_list = [None] * N
    deltas_list[-1] = activations_list[-1] - labels
    for n in range(N)[:-1]:
        m = -(n+1)
        comp1 = np.matmul(deltas_list[m], weights_list[m])
        comp2 = activations_list[m-1] * (1 - activations_list[m-1])
        deltas_list[m-1] = comp1 * comp2
    return(deltas_list)

def compute_gradients(weights_list, bias_list, inputs, activations_list, 
                      deltas_list, regulariser):
    """Compute regularised gradients for MLP weights."""
    sample_size, N = len(inputs), range(len(weights_list))
    activations_list = [inputs] + activations_list
    grads_list_weights = [None] * len(weights_list)
    grads_list_biases = [np.sum(deltas_list[n].T, 1)[:,np.newaxis]/sample_size\
                         for n in N]
    grads_list_weights = [(np.matmul(deltas_list[n].T, activations_list[n])\
                          +regulariser*weights_list[n])/sample_size for n in N]
    return [grads_list_weights, grads_list_biases]

def sigmoid_cost(outputs, labels, weights_list, regulariser):
    """Compute the cost function of a sigmoid MLP net."""
    max_val = np.finfo(outputs.dtype).max
    min_val = 1/max_val
    pos_cost = labels * np.log(np.clip(outputs, min_val, max_val))
    neg_cost = (1-labels)*np.log(np.clip(1-outputs, min_val, max_val))
    inner_cost = -np.sum(pos_cost + neg_cost)
    reg_cost = sum([np.sum(weights_list[n]**2) \
            for n in range(len(weights_list))]) * regulariser / 2
    return((inner_cost + reg_cost)/len(labels))

def update_weights(weights_list, bias_list,
                   speeds_list_weights, speeds_list_biases,
                   grads_list_weights, grads_list_biases, 
                   learning_rate, momentum):
    """Update weights from activations and deltas."""
    N = range(len(weights_list))
    # Scale gradients by learning rate
    glw = [grads_list_weights[n] * learning_rate for n in N]
    glb = [grads_list_biases[n] * learning_rate for n in N]
    # Update speeds
    slw = [speeds_list_weights[n] * momentum - glw[n] for n in N]
    slb = [speeds_list_biases[n] * momentum - glb[n] for n in N]
    # Update weights and biases
    weights_list = [weights_list[n] + slw[n] for n in N]
    bias_list = [bias_list[n] + slb[n] for n in N]
    return [weights_list, bias_list, slw, slb]

def get_starting_weights(inputs, labels, n_hidden):
    """Randomly initialise starting weights based on stated architecture."""
    architecture = [inputs.shape[1]] + n_hidden + [labels.shape[1]]
    weights_list = [None] * (len(architecture)-1)
    bias_list = [None] * (len(architecture)-1)
    for n in range(len(architecture))[:-1]:
        weights = np.random.randn(architecture[n+1], architecture[n])\
                /np.sqrt(architecture[n])
        biases = np.random.randn(architecture[n+1], 1)/np.sqrt(architecture[n])
        weights_list[n], bias_list[n] = weights, biases
    return([weights_list, bias_list])

def descend(inputs, labels, weights_list, bias_list, 
            speeds_list_weights, speeds_list_biases, 
            learning_rate, regulariser, momentum):
    """Perform one iteration of gradient descent."""
    activations_list = forward_propagation(inputs, weights_list, bias_list)
    deltas_list = backpropagation(activations_list, weights_list, labels)
    [grads_list_weights, grads_list_biases] = compute_gradients(weights_list,
            bias_list, inputs, activations_list, deltas_list, regulariser)
    [weights_list, bias_list, speeds_list_weights, speeds_list_biases] = \
            update_weights(weights_list, bias_list, speeds_list_weights, 
                           speeds_list_biases, grads_list_weights,
                           grads_list_biases, learning_rate, momentum)
    cost = sigmoid_cost(activations_list[-1], labels, weights_list,
                        regulariser)
    return([weights_list, bias_list, speeds_list_weights, speeds_list_biases,
            cost])

def descend_epoch(inputs, labels, weights_list, bias_list, speeds_init_weights,
                  speeds_init_biases, learning_rate_initial, batch_size, 
                  regulariser, momentum, cost_init, verbose):
    """Perform one epoch of gradient descent."""
    # Initialise
    sample_size = len(inputs)
    order = np.random.choice(np.arange(len(inputs)), len(inputs), False)
    inputs = inputs[order]
    labels = labels[order]
    batch = np.arange(batch_size)
    costs = np.zeros(math.ceil(sample_size/batch_size))
    # Learn from batches
    learning_rate, speeds_list_weights, speeds_list_biases = \
            learning_rate_initial, speeds_init_weights, speeds_init_biases
    n = 0
    while len(batch) > 0:
        [weights_list, bias_list, speeds_list_weights, speeds_list_biases,
         costs[n]] = descend(inputs[batch], labels[batch], weights_list, 
         bias_list, speeds_list_weights, speeds_list_biases, learning_rate,
         regulariser, momentum)
        batch += batch_size
        batch = batch[batch < sample_size]
        n += 1
    # Final cost
    outputs_final = forward_propagation(inputs, weights_list, bias_list)[-1]
    cost_final = sigmoid_cost(outputs_final, labels, weights_list, regulariser)
    if verbose: print("Final cost:", cost_final, end = ".\n")
    if cost_final >= cost_init:
        learning_rate /= 2
    return([weights_list, bias_list, speeds_list_weights, speeds_list_biases,
            cost_final, learning_rate])

def gradient_descent(inputs, labels, # Training data
        learning_rate_initial, learning_rate_min, max_epochs, # LR schedule
        n_hidden, batch_size, regulariser, momentum, # Hyperparameters
        verbose=True, profile=False):
    """Perform n complete epochs of stochastic gradient descent and return
    the best."""
    # Initialise
    p = cProfile.Profile() if profile else None
    if profile: p.enable()
    batch = np.arange(batch_size)
    weights_lists = [None] * max_epochs
    bias_lists = [None] * max_epochs
    n_batches = np.ceil(len(inputs)/batch_size).astype(int)
    costs = np.zeros([max_epochs+1])
    [weights_init, biases_init] = get_starting_weights(inputs[batch],
            labels[batch], n_hidden)
    speeds_init_weights = [np.zeros_like(w) for w in weights_init]
    speeds_init_biases = [np.zeros_like(b) for b in biases_init]
    # Compute initial cost
    outputs_init = forward_propagation(inputs, weights_init, biases_init)[-1]
    costs[0] = sigmoid_cost(outputs_init, labels, weights_init, regulariser)
    if verbose: print("   ","Initial cost: {}.".format(costs[0]))
    # Run first epoch
    def report_epoch(n, alpha, verbose):
        if verbose: print("   ","Epoch {0} (α = {1}):".format(n,alpha),
                end=" ")
    report_epoch(1, learning_rate_initial, verbose)
    [weights_lists[0], bias_lists[0], speeds_list_weights, speeds_list_biases,
     costs[1], learning_rate] = descend_epoch(inputs, labels, weights_init,
     biases_init, speeds_init_weights, speeds_init_biases,
     learning_rate_initial, batch_size, regulariser, momentum, costs[0],
     verbose)
    # Run remaining epochs
    n = 1
    while n < max_epochs and learning_rate >= learning_rate_min:
        report_epoch(n+1, learning_rate, verbose)
        [weights_lists[n], bias_lists[n], speeds_list_weights, 
         speeds_list_biases, costs[n+1], learning_rate] = descend_epoch(
         inputs, labels, weights_lists[n-1], bias_lists[n-1],
         speeds_list_weights, speeds_list_biases, learning_rate,
         batch_size, regulariser, momentum, costs[n], verbose)
        n += 1
    # Determine best output and return
    best_epoch = np.argmin(costs[costs != 0])
    print("   ", "Best epoch:", best_epoch)
    best_weights = weights_lists[best_epoch-1]
    best_biases = bias_lists[best_epoch-1]
    out = {"weights":best_weights, "biases": best_biases, "epochs_run":n-1,
           "costs":costs[:best_epoch], "best_epoch": best_epoch}
    if profile:
        p.create_stats()
        s = pstats.Stats(p)
        s.strip_dirs()
        s.sort_stats("cumtime")
        d = s.stats
        a = np.array([k[2] for k in d.keys()])
        epoch_stats = d[list(d.keys())[np.nonzero(a == "descend_epoch")\
                        [0][0]]]
        epoch_time = epoch_stats[3]/epoch_stats[0]
        print("Total time (s):", round(s.total_tt, 2))
        print("Time per epoch (s):", round(epoch_time,2))
        out["profile_stats"] = s
    return(out)
    # TODO: Am I returning the right weights list here?

def train_hyperparameters(inputs_train, labels_train, # Training data
        inputs_val, labels_val, # Validation data
        learning_rate_initial, learning_rate_min, max_epochs, # LR schedule
        n_hidden_vals, batch_size_vals, regulariser_vals, momentum_vals, # Hyperparameters
        verbose = True, profile = False):
    """Train MLP using various hyperparameter values and pick the best ones
    using the validation dataset."""
    combs = itertools.product(batch_size_vals, regulariser_vals, n_hidden_vals,
            momentum_vals)
    cost = math.inf
    n = 0
    for batch_size,regulariser,n_hidden,momentum in combs:
        print("\nHyperparameters (batch size/regulariser/neurons/momentum):",
                "{0} {1} {2} {3}".format(batch_size,regulariser,n_hidden,momentum))
        nn = gradient_descent(inputs_train, labels_train, learning_rate_initial,
                learning_rate_min, max_epochs, n_hidden, batch_size,
                regulariser, momentum, verbose, profile)
        output_val = forward_propagation(inputs_val, nn["weights"],
                                         nn["biases"])[-1]
        cost_val = sigmoid_cost(output_val, labels_val, nn["weights"], 0)
        print("Validation error:", cost_val)
        if cost_val < cost:
            out = {"batch_size":batch_size,
                    "regulariser":regulariser, "weights":nn["weights"],
                    "biases": nn["biases"],
                    "trace":nn["costs"], "n_hidden":n_hidden,
                    "epochs_run":nn["epochs_run"], "momentum":momentum}
            cost = cost_val
    print("\nBest hyperparameters (batch size/regulariser/neurons/momentum):",
            "{0} {1} {2} {3}".format(out["batch_size"],out["regulariser"],
                out["n_hidden"],out["momentum"]))
    if verbose: print("Best validation error:", cost)
    return out
