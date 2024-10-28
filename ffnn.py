import math
import numpy as np

"""
Implementation of simple feed forward neural network
"""

# activation function
def ReLU(x):
    with np.nditer(x, op_flags=['readwrite']) as it:
        for y in it:
            y[...] = max(0, y)


def sigmoid(x):
    with np.nditer(x, op_flags=['readwrite']) as it:
        for y in it:
            y[...] =  1 / ( 1 + math.e ** -y)


def softmax(x):
    with np.nditer(x, op_flags=['readwrite']) as it:
        for y in it:
            y[...] = math.e ** y / (np.sum(np.pow(np.repeat([math.e], 3), x)))


def FFNN(input_vector, mode="binary"):
    print(input_vector)
    # w1
    w_1_1 = np.array([0.5, 0.1, -0.8, 0.1])
    w_1_2 = np.array([0.9, -0.1, 0.4, -0.4])
    b_1 = np.array([0.05, 0.01])
    h_1 = np.add(np.array([np.dot(input_vector, w_1_1), np.dot(input_vector, w_1_2)]), b_1)
    print(h_1)
    # w2
    ReLU(h_1)
    w_2_1 = np.array([0.8, -0.7])
    w_2_2 = np.array([0.2, -0.05])
    b_2 = np.array([0.3, 0.3])
    h_2 = np.add(np.array([np.dot(h_1, w_2_1), np.dot(h_1, w_2_2)]), b_2)
    print(h_2)
    # w3
    ReLU(h_2)
    # return
    if (mode == "binary"):
        w_3 = np.array([0.9, -0.4])
        b_3 = np.array([-0.01])
        y = np.add(np.dot(h_2, w_3), b_3)
        sigmoid(y)
    elif(mode == "multiclass"):
        w_3_1 = np.array([0.9, -0.4])
        w_3_2 = np.array([-0.3, 0.2])
        w_3_3 = np.array([-0.3, 0.8])
        b_3 = np.array([-0.01, 0.1, 0.05])
        y_1 = np.dot(h_2, w_3_1)
        y_2 = np.dot(h_2, w_3_2)
        y_3 = np.dot(h_2, w_3_3)
        y = np.add(np.array([y_1, y_2, y_3]), b_3)
        softmax(y)
    return y


print(FFNN(np.array([ 1, 0, 28, 4.5])))
print(FFNN(np.array([ 1, 0, 28, 4.5]), "multiclass"))

