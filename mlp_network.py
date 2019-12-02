#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 09:29:27 2019

@author: will
"""

import torch, torch.nn as nn, torch.optim as optim

class MLP(nn.Module):
    
    def __init__(self, n_neurons):
        super(MLP, self).__init__()
        self.n_layers = len(n_neurons)
        for n in range(self.n_layers)[:-1]:
            attr_name = "fc" + str(n+1)
            attr_val = nn.Linear(n_neurons[n], n_neurons[n+1])
            setattr(self, attr_name, attr_val)
            
    def forward(self, inputs):
        x = inputs
        for n in range(self.n_layers)[:-1]:
            f = getattr(self, "fc" + str(n+1))
            x = torch.sigmoid(f(x))
        return(x)
        
    def loss(self, inputs, outputs):
        return nn.BCELoss()(self.forward(inputs), outputs)