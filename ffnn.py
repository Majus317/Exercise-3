#!/usr/bin/env python3

"""
Implementation of simple feed forward neural network
"""

import numpy as np

# Input Vector (Example)
input_vector = np.array([1, 0, 28, 4.5])

# Network 1 and Network 2: w1 and w2 are the same
# Weight matrix between input vector 1 and hidden layer 1 
w1_matrix1 = np.array([0.5, 0.1, -0.8, 0.1])
w1_matrix2 = np.array([0.9, -0.1, 0.4, -0.4])
h1_bias = np.array([0.05, 0.01])
# Weight matrix between hidden layer 1 and hidden layer 2
w2_matrix1 = np.array([0.8, 0.2])
w2_matrix2 = np.array([-0.7, 0.3])
h2_bias = np.array([0.3, 0.3])

# Network 1
# Weight matrix between hidden layer 2 and hidden layer 3
n1_w3_matrix = np.array([0.9, -0.4])
n1_h3_bias = -0.01

# Network 2
# Weight matrix between hidden layer 2 and hidden layer 3
n2_w3_matrix1 = np.array([0.9, -0.4])
n2_w3_matrix2 = np.array([-0.3, 0.2])
n2_w3_matrix3 = np.array([-0.3, 0.8])
n2_h3_bias = np.array([-0.01, 0.1, 0.05])

# activation function for all hidden layers in both networks
def ReLU(x):
    return np.maximum(0, x)

# activation function for the output layer of network 1
def sigmoid(x):
    x = 1 / (1 + np.exp(-x))
    return x

# activation function for the output layer of network 2
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def FFNN(input_vector, mode="binary"):
    # Dot product of input vector and hidden layer 1
    l1_m1 = np.dot(input_vector, w1_matrix1)
    l1_m2 = np.dot(input_vector, w1_matrix2)
    w1_l1 = np.array([l1_m1, l1_m2])
    bias_l1 = np.add(w1_l1, h1_bias) # add bias
    hidden_layer1 = ReLU(bias_l1)

    # Dot product of hidden layer 1 and hidden layer 2
    l2_m1 = np.dot(hidden_layer1, w2_matrix1)
    l2_m2 = np.dot(hidden_layer1, w2_matrix2)
    w2_l2 = np.array([l2_m1, l2_m2])
    bias_l2 = np.add(w2_l2, h2_bias) # add bias
    hidden_layer2 = ReLU(bias_l2)

    if (mode == "binary"):
        l3_m = np.dot(hidden_layer2, n1_w3_matrix)
        bias_l3 = np.add(l3_m, n1_h3_bias) # add bias 
        n1_hidden_layer3 = sigmoid(bias_l3)
        return n1_hidden_layer3
    elif (mode == "multiclass"):
        l3_m1 = np.dot(hidden_layer2, n2_w3_matrix1)
        l3_m2 = np.dot(hidden_layer2, n2_w3_matrix2)
        l3_m3 = np.dot(hidden_layer2, n2_w3_matrix3)
        w3_l3 = np.array([l3_m1, l3_m2, l3_m3])
        bias_l3_n2 = np.add(w3_l3, n2_h3_bias) # add bias
        n2_hidden_layer3 = softmax(bias_l3_n2)
        return n2_hidden_layer3
    else:
        print("Mode is invalid!")
        return None

# Testing
print("Binary classification output:", FFNN(input_vector, mode="binary"))
print("Multiclass classification output:", FFNN(input_vector, mode="multiclass"))

    

    

