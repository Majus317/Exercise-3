#!/usr/bin/env python3

"""
Implementation of simple feed forward neural network
"""

import numpy as np


def ReLU(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x, axis=0)


def FFNN(input_vector, mode="binary"):
    # Layer 1: Hidden layer 1 weights (4 inputs, 2 neurons)
    W1 = np.array([[0.5, 0.1, -0.8, 0.1],
                   [0.9, -0.1, 0.4, -0.4]])
    b1 = np.array([0.05, 0.01])

    # Layer 2: Hidden layer 2 weights (2 neurons, 2 inputs from layer 1)
    W2 = np.array([[0.8, 0.7],
                   [0.2, 0.3]])
    b2 = np.array([0.05, 0.03])

    # Forward pass, Hidden layer 1 with ReLU activation
    z1 = np.dot(W1, input_vector) + b1
    a1 = ReLU(z1)

    # Forward pass, Hidden layer 2 with ReLU activation
    z2 = np.dot(W2, a1) + b2
    a2 = ReLU(z2)

    if mode == "binary":
        # For network 1 (Binary classification)
        W3 = np.array([[0.9, -0.4]])  # 1 output neuron
        b3 = -0.01

        # Output layer (binary classification)
        z3 = np.dot(W3, a2) + b3
        output = sigmoid(z3)
        return output

    elif mode == "multiclass":
        # For network 2 (Multiclass classification)
        W3 = np.array([[0.9, -0.3],
                       [-0.4, 0.2],
                       [0.3, 0.8]])  # 3 output neurons
        b3 = np.array([-0.01, 0.1, 0.05])  # Bias for 3 output neurons

        # Output layer (multiclass classification)
        z3 = np.dot(W3, a2) + b3
        output = softmax(z3)
        return output


if __name__ == "__main__":
    input_vector = np.array([1.0, 2.0, 3.0, 4.0])  # Example input vector (4 components)

    # Test binary classification mode
    binary_output = FFNN(input_vector, mode="binary")
    print("Binary classification output (probability):", binary_output)

    # Test multiclass classification mode
    multiclass_output = FFNN(input_vector, mode="multiclass")
    print("Multiclass classification output (probabilities):", multiclass_output)
