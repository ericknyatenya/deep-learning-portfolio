"""
Simple neural network layer primitives (NumPy-based) for educational purposes.
"""

import numpy as np


class Dense:
    """A simple fully-connected layer."""
    def __init__(self, in_features, out_features, lr=0.01):
        self.W = np.random.randn(out_features, in_features) * 0.01
        self.b = np.zeros((out_features, 1))
        self.lr = lr
        self.cache = None

    def forward(self, x):
        z = self.W.dot(x) + self.b
        self.cache = x
        return z

    def backward(self, dz):
        x = self.cache
        m = x.shape[1]
        dW = (1 / m) * dz.dot(x.T)
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        dx = self.W.T.dot(dz)

        # gradient descent update
        self.W -= self.lr * dW
        self.b -= self.lr * db

        return dx
