import numpy as np
import matplotlib.pyplot as plt
import time
from data_generator import *


def batch_gradient_descent(X, Y, epochs=100, learning_rate=0.01):
    m, n = X.shape
    X = np.c_[np.ones((m, 1)), X]
    theta = np.random.randn(n + 1, 1)

    losses = []

    for epoch in range(epochs):
        predictions = X.dot(theta)
        errors = predictions - Y
        gradients = 2/m*X.T.dot(errors)
        theta -= learning_rate*gradients
        loss = compute_loss(Y, predictions)
        losses.append(loss)
    
    return theta, losses