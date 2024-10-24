import numpy as np

"""
Implementation of simple feed forward neural network
"""

def ReLU(x):
    if x <= 0:
        x=0
    return x 


def sigmoid(x):
    x = 1/(1+ np.exp(-x))
    return x


def softmax(x):
    x = np.exp(x)/sum(np.exp(x))
    print(x)
    return x


def FFNN(input_vector, mode="binary"):
    pass
