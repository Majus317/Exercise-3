#!/usr/bin/env python3
import numpy as np


def ReLU(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


def FFNN(input_vector, mode="binary"):

    h1_weights = np.array([[0.5, 0.1, -0.8, 0.1], 
                           [0.9, -0.1, 0.4, -0.4]])
    h1_bias = np.array([0.05, 0.01])
    hidden_layer1 = ReLU(np.dot(h1_weights, input_vector) + h1_bias)

 
    h2_weights = np.array([[0.8, -0.7], 
                           [0.2, 0.3]])
    h2_bias = np.array([-0.05, 0.3])
    hidden_layer2 = ReLU(np.dot(h2_weights, hidden_layer1) + h2_bias)


    if mode == "binary":
        output_weights = np.array([-0.9, -0.4])
        output_bias = -0.01
        output = sigmoid(np.dot(output_weights, hidden_layer2) + output_bias)
        return output
    elif mode == "multiclass":
        output_weights = np.array([[0.9, -0.4], 
                                   [-0.3, 0.2], 
                                   [-0.3, 0.8]])
        output_bias = np.array([-0.01, 0.1, 0.05])
        output = softmax(np.dot(output_weights, hidden_layer2) + output_bias)
        return output
    else:
        print("Invalid mode")
        return None



v = np.array([1, 0, 28, 4.5])
v2 = np.array([1.4, 34.6, -17.3, 2.5])
print("Vector1 Binary:", FFNN(v, "binary"))
print("Vector1 Multiclass:", FFNN(v, "multiclass"))
print("Vector2 Binary:", FFNN(v2, "binary"))
print("Vector2 Multiclass:", FFNN(v2, "multiclass"))
