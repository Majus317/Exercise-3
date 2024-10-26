#!/usr/bin/env python3

"""
Implementation of simple feed forward neural network
"""

import numpy as np


def ReLU(x):
    for i in range(len(x)):
        if x[i] < 0:
            x[i] = 0
    return x


def sigmoid(x):
    x = 1/(1+ np.exp(-x))
    return x


def softmax(x):
    #return np.exp(x) / np.sum(np.exp(x), axis=0)
    #return x

    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def FFNN(input_vector, mode="binary"):

    #biasses
    b1 = np.array([0.05, 0.01])
    b2 = np.array([-0.05, 0.3])
    b3b = -0.01
    b3m = np.array([-0.01, 0.1, 0.05])

    #hidden layer 1
    h1_1 = np.dot(input_vector, np.array([0.5, 0.1, -0.8, 0.1]))
    h1_2 = np.dot(input_vector, np.array([0.9, -0.1, 0.4, -0.4]))

    h1 = np.array([h1_1, h1_2])

    #add bias
    h1 = np.add(h1, b1)
    
    #ReLU
    y = ReLU(h1)

    #print(h1)

    #hidden layer 2
    h2_1 = np.dot(h1, np.array([0.8, 0.2]))
    h2_2 = np.dot(h1, np.array([-0.7, 0.3]))

    h2 = np.array([h2_1, h2_2])

    print(h2)

    #add bias
    h2 = np.add(h2, b2)

    #print(h2)

    #ReLU
    y = ReLU(h2)

    #print(h2)

    #output layer
    if mode == "binary":
        y = np.dot(h2, np.array([0.9, -0.4]))

        #output bias 
        y = np.add(y, b3b)

        #output sigmoid
        y = sigmoid(y)

        return y
        
    elif mode == "multiclass":
        y1 = np.dot(h2, np.array([0.9, -0.4]))
        y2 = np.dot(h2, np.array([-0.3, 0.2]))
        y3 = np.dot(h2, np.array([-0.3, 0.8]))

        y = np.array([y1, y2, y3])

        #output bias 
        y = np.add(y, b3m)

        #output softmax
        y = softmax(y)
    
        return y
    

print(FFNN( np.array([1, 0, 28, 4.5]), "binary"))
print(FFNN( np.array([0.4, 13, -9.1, 6.5]), "multiclass"))
