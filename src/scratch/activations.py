"""
Common activation functions and their derivatives.
"""

import numpy as np


def sigmoid(z):
    a = 1 / (1 + np.exp(-z))
    return a


def sigmoid_derivative(z):
    a = sigmoid(z)
    return a * (1 - a)


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)
