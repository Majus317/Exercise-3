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
    x = 1/(1 + np.exp(-x))
    return x

def softmax(x):
    x = x.astype(float)
    sum = np.sum(x)
    for i in range(len(x)):
        x[i] = x[i]/sum
    return x



def FFNN(input_vector, mode="binary"):
    h1n1 = np.dot(input_vector, np.array([0.5, 0.1, -0.8, 0.1]))
    h1n2 = np.dot(input_vector, np.array([0.9, -0.1, 0.4, -0.4]))
    hiddenlayer1combined = np.array([h1n1, h1n2])
    bias1 = np.array([0.05, 0.01])
    hiddenlayer1combined = np.add(hiddenlayer1combined, bias1)
    hiddenlayer1 = ReLU(hiddenlayer1combined)
    # print("Hiddenlayer1 ",  hiddenlayer1)
    h2n1 = np.dot(hiddenlayer1, np.array([0.8, -0.7]))
    h2n2 = np.dot(hiddenlayer1, np.array([0.2, 0.3]))
    bias2 = np.array([-0.05, 0.3])
    hiddenlayer2combined = np.array([h2n1, h2n2])
    hiddenlayer2combined = np.add(hiddenlayer2combined, bias2)
    hiddenlayer2 = ReLU(hiddenlayer2combined)
    # print("Hiddenlayer2 ", hiddenlayer2)
    if(mode == "binary"):
        output = np.dot(hiddenlayer2, np.array([-0.9, -0.4]))
        bias3 = -0.01
        output = output + bias3
        output = sigmoid(output)
        return output
    elif(mode == "multiclass"):
        outputn1 = np.dot(hiddenlayer2, np.array([0.9, -0.4]))
        outputn2 = np.dot(hiddenlayer2, np.array([-0.3, 0.2]))
        outputn3 = np.dot(hiddenlayer2, np.array([-0.3, 0.8]))
        bias3 = np.array([-0.01, 0.1, 0.05])
        output = np.array([outputn1, outputn2, outputn3])
        output = np.add(output, bias3)
        output = softmax(output)
        return output
    else:
        print("Invalid mode")
        return None

v = np.array([1, 0, 28, 4.5])
v2 = np.array([1.4, 34.6, -17.3, 2.5])
print("Vektor1 Binary ", FFNN(v , "binary"))
print("Vektor1 Multiclass ", FFNN(v , "multiclass"))
print("Vektor2 Binary ", FFNN(v2 , "binary"))
print("Vektor2 Multiclass ", FFNN(v2 , "multiclass"))
