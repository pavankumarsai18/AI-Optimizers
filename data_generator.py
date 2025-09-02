import numpy as np
import matplotlib.pyplot as plt
import time

def generate_data(num_samples):
    np.random.seed(31)
    # np.random.rand(num_samples, 1) --> generates random numbers with mean=0, std_dev = 1
    X = np.random.rand(num_samples, 1) * 10
    Y = 3*X + 7 + np.random.randn(num_samples, 1)
    return {"x": X, "y": Y}


def compute_loss(Y, predictions):
    return np.mean((Y - predictions)**2)
