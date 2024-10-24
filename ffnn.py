import numpy as np

"""
Implementation of simple feed forward neural network
"""

def ReLU(x):
    return np.maximum(0,x)


def sigmoid(x):
    x = 1/(1+ np.exp(-x))
    return x


def softmax(x):
    x = np.exp(x)/sum(np.exp(x))
    print(x)
    return x

def FFNN_bi(x):
    #wheights for binary network
    W1 = np.array([[0.5, 0.1], [-0.8, -0.1], [0.9, 0.1], [0.4, 0.05]])
    W2 = np.array([[0.8, -0.7], [0.2, 0.3]])
    W3 = np.array([0.9, -0.4])
    #biases for binary network
    b1 = np.array([0.01, -0.05])
    b2 = np.array([0.1, 0.3])
    b3 = np.array([-0.01])

    #Layer 1
    h1 = ReLU(np.dot(x, W1) + b1)
    #Layer 2
    h2 = ReLU(np.dot(h1, W2) + b2)
    #Output Layer 
    y = sigmoid(np.dot(h2, W3) + b3)
    return y

def FFNN_mult(x):
    pass

def FFNN(input_vector, mode):
    if mode == "binary":
        print(FFNN_bi(input_vector))
    elif mode == "multiclass":
        print(FFNN_mult(input_vector))
    else:
        print("Invalid input! The only valid inputs are 'binary' and 'multiclass'")



FFNN(np.array([2, 0.5, 1, 0.2]),"binary")