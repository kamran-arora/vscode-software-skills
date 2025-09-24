from functions import *
import numpy as np


def test_add():
    assert add(1, 2) == 3


def test_matmul():
    mat = np.array([[1, 0], [0, 1]])
    vec = np.array([1, 2])
    assert np.allclose(mat_mul(mat, vec), np.array([1, 2]))
