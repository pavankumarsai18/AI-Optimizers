import numpy as np
import matplotlib.pyplot as plt
import time
from data_generator import *

def stochastic_gradient_descent(X, Y, epochs=100, learning_rate=0.01):
    m, n = X.shape
    X = np.c_[np.ones((m, 1)), X]
    theta = np.random.randn(n + 1, 1)

    losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        random_index = np.random.randint(m)
        xi = X[random_index:random_index+1]
        yi = Y[random_index:random_index+1]
        predictions = xi.dot(theta)
        errors = predictions - yi
        gradients = 2/m*xi.T.dot(errors)
        theta -= learning_rate*gradients
        epoch_loss += compute_loss(Y, predictions)
        losses.append(epoch_loss/m)
    
    return theta, losses