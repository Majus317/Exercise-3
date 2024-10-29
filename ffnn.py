#!/usr/bin/env python3

"""
Implementation of simple feed forward neural network
"""

import numpy as np


def ReLU(x):
    return np.max(x, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=0)


def FFNN(input_vector, mode):
    # weights
    W1_h1 = np.array([0.5, 0.1, -0.8, 0.1])
    W1_h2 = np.array([0.9, -0.1, 0.4, -0.4])
    W2_h1 = np.array([0.8, 0.2])
    W2_h2 = np.array([-0.7, 0.3])
    W3_N1_y = np.array([0.9, -0.4])
    W3_N2_y1 = np.array([0.9, -0.4])
    W3_N2_y2 = np.array([-0.3, 0.2])
    W3_N2_y3 = np.array([-0.3, 0.8])
    # bias
    b1 = np.array([0.05, 0.01])
    b2 = np.array([-0.05, 0.3])
    b3_N1 = np.array([-0.01])
    b3_N2 = np.array([-0.01, 0.1, 0.05])

    # calculation hiddenlayer 1
    h1_1 = np.dot(input_vector, W1_h1)
    h2_1 = np.dot(input_vector, W1_h2)
    hl1 = ReLU(np.add(np.array([h1_1, h2_1]), b1))
    # calculation hiddenlayer 2
    h1_2 = np.dot(hl1, W2_h1)
    h2_2 = np.dot(hl1, W2_h2)
    hl2 = ReLU(np.add(np.array([h1_2, h2_2]), b2))

    if mode == "binary":
        y_N1 = np.dot(hl2, W3_N1_y)
        y = sigmoid(np.add(y_N1, b3_N1))
        print("Output layer in binary mode: ", y)
    elif mode == "multiclass":
        y1 = np.dot(hl2, W3_N2_y1)
        y2 = np.dot(hl2, W3_N2_y2)
        y3 = np.dot(hl2, W3_N2_y3)
        y = softmax(np.add(np.array([y1, y2, y3]), b3_N2))
        print("Output layer in multiclass mode: ", y)
    else:
        print("Mode is not avaible!")


FFNN(np.array([4, 0.6, -1.2, 3.2]), "binary")
FFNN(np.array([4, 0.6, -1.2, 3.2]), "multiclass")
