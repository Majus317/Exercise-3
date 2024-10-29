#!/usr/bin/env python3

"""
Implementation of simple feed forward neural network
"""

import numpy as np

#weights
w1 = np.matrix([[0.5, 0.1], [-0.8, 0.1],[0.9, -0.1], [0.4, -0.4]])
w2 = np.matrix([[0.8, 0.2],[-0.7, 0.3]])
w3 = np.matrix([[0.9],[0.4]])
#biases
b1 = np.matrix([[0.05], [0.01]])
b2 = np.matrix([[-0.05],[0.3]])
b3 = np.matrix([[-0.01]])

def ReLU(x):
    return np.maximum(0,x)

def sigmoid(x):
    return 1/(1+np.exp(-x))

#oof unsure
def softmax(x):
    eho = np.exp(x)
    return eho/sum(eho)

def FFNN(input_vector, mode="binary"):
    x = input_vector
#this is where the maths should be mathing but instead... uh i need to come back to this
    
    hl1 = ReLU(np.dot(x, w1)+b1)
    x = hl1
    hl2 = ReLU(np.dot(x, w2)+b2)
    out = np.dot(hl2, w3)+b3

    if mode == "binary":
        return sigmoid(out)
    elif mode == "multiclass":
        return softmax(out)
#hm
input = np.matrix([0,2,1,4])
FFNN(input, mode = "binary")