#!/usr/bin/env python3

import numpy as np

def ReLU(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x))  
    return exp_x / exp_x.sum(axis=0)

def FFNN(input_vector, mode="binary"):
    x = np.array(input_vector)

    W1_1 = np.array([[0.5, 0.1], [-0.8, 0.9], [0.5, 0.1], [0.4, 0.05]])
    b1_1 = 0.1
    W2_1 = np.array([[0.8, -0.05], [0.2, 0.3]])
    b2_1 = np.array([0.1, -0.1])
    W3_1 = np.array([[0.9, -0.4]])
    b3_1 = -0.01

    W1_2 = W1_1
    b1_2 = b1_1
    W2_2 = W2_1
    b2_2 = b2_1
    W3_2 = np.array([[0.9, -0.3, 0.05], [-0.3, 0.4, -0.2]])
    b3_2 = np.array([-0.01, 0.1, -0.05])

    if mode == "binary":
        h1 = ReLU(np.dot(x, W1_1) + b1_1)  
        h2 = ReLU(np.dot(h1, W2_1) + b2_1)        
        y = sigmoid(np.dot(h2, W3_1) + b3_1)     
        return y.item() 
    elif mode == "multiclass":
        h1 = ReLU(np.dot(x, W1_2) + b1_2)         
        h2 = ReLU(np.dot(h1, W2_2) + b2_2) 
        y = softmax(np.dot(h2, W3_2) + b3_2) 
        return y 
    else:
        raise ValueError("Mode should be 'binary' or 'multiclass'")

input_vector = [1, 1, 1, 1]  
output_binary = FFNN(input_vector, mode="binary")
output_multiclass = FFNN(input_vector, mode="multiclass")

print("Output for Network 1 (Binary):", output_binary)
print("Output for Network 2 (Multiclass):", output_multiclass)