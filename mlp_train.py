import torch, math, itertools, cProfile, pstats, numpy as np
from mlp_network import *

def descend_epoch(inputs, labels, network, # Data and network
                  learning_rate, batch_size, regulariser, momentum, # Hypers
                  cost_init, verbose):
    """Perform one epoch of gradient descent."""
    # Initialise
    sample_size = len(inputs)
    order = torch.randperm(len(inputs))
    inputs = inputs[order]
    labels = labels[order]
    batch = torch.arange(batch_size)
    costs = torch.zeros(math.ceil(sample_size/batch_size))
    optimiser = optim.SGD(network.parameters(), lr=learning_rate,
                          momentum=momentum, weight_decay=regulariser)
    # Learn from batches
    n = 0
    while len(batch) > 0:
        optimiser.zero_grad()
        loss = network.loss(inputs[batch], labels[batch])
        loss.backward()
        optimiser.step()
        batch += batch_size
        batch = batch[batch < sample_size]
        n += 1
    # Final cost
    cost_final = float(network.loss(inputs, labels))
    if verbose: print("Final cost:", cost_final, end = ".\n")
    if cost_final >= cost_init:
        learning_rate /= 2
    return([network, cost_final, learning_rate])

def gradient_descent(inputs, labels, # Training data
        learning_rate_initial, learning_rate_min, max_epochs, # LR schedule
        n_hidden, batch_size, regulariser, momentum, # Hyperparameters
        verbose=True, profile=False):
    """Perform n complete epochs of stochastic gradient descent and return
    the best."""
    # Initialise
    p = cProfile.Profile() if profile else None
    if profile: p.enable()
    costs = torch.zeros([max_epochs+1])
    learning_rates = torch.zeros([max_epochs+1])
    networks = [None] * (max_epochs+1)
    n_neurons = [inputs.shape[1]] + n_hidden + [labels.shape[1]]
    networks[0] = MLP(n_neurons)
    learning_rates[0] = learning_rate_initial
    costs[0] = networks[0].loss(inputs, labels)
    if verbose: print("   ","Initial cost: {}.".format(costs[0]))
    # Run epochs
    def report_epoch(n, alpha, verbose):
        if verbose: print("   ","Epoch {0} (α = {1}):".format(n,alpha),
                end=" ")
    n = 1
    while n <= max_epochs and learning_rates[n-1] >= learning_rate_min:
        report_epoch(n, learning_rates[n-1], verbose)
        [networks[n], costs[n], learning_rates[n]] = descend_epoch(
                inputs, labels, networks[n-1], learning_rates[n-1],
                batch_size, regulariser, momentum, costs[n-1], verbose)
        n += 1
    # Determine best output and return
    best_epoch = int(torch.argmin(costs[costs != 0]))
    print("   ", "Best epoch:", best_epoch)
    best_network = networks[best_epoch]
    out = {"network": best_network, "epochs_run":n-1,
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

def train_hyperparameters(inputs_train, labels_train, # Training data
        inputs_val, labels_val, # Validation data
        learning_rate_initial, learning_rate_min, max_epochs, # LR schedule
        n_hidden_vals, batch_size_vals, regulariser_vals, momentum_vals, # Hyperparameters
        verbose = True, profile = False, max_tries = 10):
    """Train MLP using various hyperparameter values and pick the best ones
    using the validation dataset."""
    combs = itertools.product(batch_size_vals, regulariser_vals, n_hidden_vals,
            momentum_vals)
    cost = math.inf
    n = 0
    for batch_size,regulariser,n_hidden,momentum in combs:
        best_epoch = 0
        n_tries = 0
        while best_epoch == 0 and n_tries < max_tries:
            print("\nHyperparameters (batch size/regulariser/neurons/momentum):",
                "{0} {1} {2} {3}".format(batch_size,regulariser,n_hidden,momentum))
            nn = gradient_descent(inputs_train, labels_train, learning_rate_initial,
                learning_rate_min, max_epochs, n_hidden, batch_size,
                regulariser, momentum, verbose, profile)
            best_epoch = nn["best_epoch"]
            if best_epoch == 0:
                print("\nRun failed! Trying again...")
        cost_val = float(nn["network"].loss(inputs_val, labels_val))
        print("Validation error:", cost_val)
        if cost_val < cost:
            out = {"batch_size":batch_size,
                    "regulariser":regulariser, "network":nn["network"],
                    "trace":nn["costs"], "n_hidden":n_hidden,
                    "epochs_run":nn["epochs_run"], "momentum":momentum}
            cost = cost_val
    print("\nBest hyperparameters (batch size/regulariser/neurons/momentum):",
            "{0} {1} {2} {3}".format(out["batch_size"],out["regulariser"],
                out["n_hidden"],out["momentum"]))
    if verbose: print("Best validation error:", cost)
    return out