#!/usr/bin/env python3

"""
Implementation of simple feed forward neural network
"""

import numpy as np
x = np.array([3,1,0.2])
def ReLU(x):
    for i in range(len(x)):
        if x[i] < 0:
            x[i] = 0
    return x

def sigmoid(x):
    x = 1/(1 + np.exp(-x))
    return x

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def FFNN(input_vector, mode="binary"):
    h1_1 = np.dot(input_vector, np.array([0.5,0.1,-0.8,0.1]))
    h1_2 = np.dot(input_vector, np.array([0.9,-0.1,0.4,-0.4]))

    l1v = np.array([h1_1,h1_2])
    l1vbias = np.array(np.add(l1v, np.array([0.05,0.01])))
    #print(l1vbias)
    hiddenlayer1 = ReLU(l1vbias)
    print(hiddenlayer1)

    h2_1 = np.dot(hiddenlayer1, np.array([0.8,0.2]))
    h2_2 = np.dot(hiddenlayer1, np.array([-0.7,0.3]))

    l2v = np.array(h2_1,h2_2)
    l2vbias = np.array(np.add(l2v,np.array([-0.05,0.3])))
    #print(l2vbias)
    hiddenlayer2 = ReLU(l2vbias)
    print(hiddenlayer2)

    if mode == "binary":
        y = np.dot(hiddenlayer2, np.array([0.9,-0.4]))
        #print(y)
        ybias = np.add(y,-0.01)
        #print(ybias)
        output = sigmoid(ybias)
        return output
    elif mode == "multiclass":
        y1 = np.dot(hiddenlayer2, np.array([0.9,-0.4]))
        y2 = np.dot(hiddenlayer2, np.array([-0.3,0.2]))
        y3 = np.dot(hiddenlayer2, np.array([-0.3,0.8]))

        y = np.array([y1,y2,y3])
        #print(y)
        ybias = np.add(y, np.array([-0.01,0.1,0.05]))
        #print(ybias)
        output = softmax(ybias)
        return output
    else:
        print("error")

v1 = np.array([1,0,28,4.5])
v2 = np.array([0.4,13,-9.1,6.5])
print(FFNN(v1, "binary"))
print(FFNN(v1,"multiclass"))
print(FFNN(v2, "binary"))
print(FFNN(v2,"multiclass"))