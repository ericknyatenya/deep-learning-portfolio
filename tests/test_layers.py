from src.scratch.layers import Dense
import numpy as np


def test_dense_forward_backward_shape():
    # small deterministic test using fixed seed
    np.random.seed(0)
    layer = Dense(in_features=3, out_features=2, lr=0.1)
    x = np.random.randn(3, 4)  # 4 examples
    z = layer.forward(x)
    assert z.shape == (2, 4)

    # backprop with ones should return dx with correct shape
    dz = np.ones_like(z)
    dx = layer.backward(dz)
    assert dx.shape == (3, 4)
