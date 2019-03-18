#!/usr/bin/env python
# -*- coding_utf-8 -*-
import numpy as np
import pandas as pd
import warnings
from sklearn import datasets
warnings.filterwarnings("ignore")

digits = datasets.load_digits()
np.random.seed(1222)

def sigmoid(x):
    return 1.0 / (1+np.exp(-x))


def init_params(n_input, n_hidden, n_output):
    """
    Params:

        n_input : size of input layer
        n_output : size of output layer
        n_hidden : size of hidden layer

        W1 : weight from input to hidden layer
        b1 : bias from input to hidden layer
        W2 : weight from hidden to output layer
        b2 : bias from hidden to output layer

    Return:

        Dictionary type {W1,b1,W2,b2}

    """

    W1 = np.random.randn(n_hidden, n_input) * 0.01
    b1 = np.zeros((n_hidden, 1))
    W2 = np.random.randn(n_output, n_hidden) * 0.01
    b2 = np.zeros((n_output, 1))

    params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return params



def forward(x, params):
    """
    Params:

        x :  training data
        param : containging W1,b1,W2,b2

    Return:

        cache : store the input and ouput of hidden_layer and output_layer

    """
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    Z1 = np.dot(W1, x) + b1
    A1 = np.tanh(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}
    return cache

def regularization(lambda_, params):
    return lambda_ * (np.sum(np.power(params['W1'],2)) +  np.sum(np.power(params['W2'],2)))


def cross_entropy(A2, y, batch, lambda_, params):
    """
    Params :
        y : training label
        A2 : output prob of output layer

    Return :
        cost : cross entropy

    """

    loss = np.multiply(np.log(A2), y) + np.multiply(np.log(1 - A2), (1 - y))
    cost = -(1.0 / batch) * np.sum(loss) + regularization(lambda_, params)
    cost = np.squeeze(cost)
    return cost


def backprop(x, y, params, cache, batch):
    """
    Params :
        x :  training data feature
        y : training data label

        param : containging W1,b1,W2,b2
        cache : store the input and ouput of hidden_layer and output_layer

    Return :
        grad : gradient descent from back to front
    """

    A1 = cache['A1']
    A2 = cache['A2']
    Z1 = cache['Z1']
    Z2 = cache['Z2']

    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    dZ2 = (A2 - y)
    dW2 = (1.0 / batch) * np.dot(dZ2, A1.T)
    db2 = (1.0 / batch) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = (1.0 / batch) * np.dot(dZ1, x.T)
    db1 = (1.0 / batch) * np.sum(dZ1, axis=1, keepdims=True)

    grad = {'dW1': dW1,
            'db1': db1,
            'dW2': dW2,
            'db2': db2}
    return grad


def update(params, grad, lr):
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    dW1 = grad['dW1']
    db1 = grad['db1']
    dW2 = grad['dW2']
    db2 = grad['db2']

    W1 = W1 - lr * dW1
    b1 = b1 - lr * db1
    W2 = W2 - lr * dW2
    b2 = b2 - lr * db2
    param = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    return param

def evaluate(x_test,y_test,params,threshold):
    pred = forward(x_test, params)['A2']
    pred[pred>=threshold] = 1
    pred[pred<threshold] = 0

    return np.sum(pred==y_test) / x_test.shape[1]


def train(x, y, x_val, y_val, n_hidden, classes, lr, threshold, lambda_):
    n_input = x.shape[0]
    batch = x.shape[1]
    params = init_params(n_input, n_hidden, classes)
    for i in range(20000):
        cache = forward(x, params)
        cost = cross_entropy(cache['A2'], y, batch, lambda_, params)
        grad = backprop(x, y, params, cache, batch)
        params = update(params, grad, lr)
        if i % 2000 == 0:
            pred = evaluate(x_val, y_val, params, threshold)
            print("Accuracy: %.4f, loss: %.6f" % (pred, cost))

if __name__ == '__main__':
    data_train_X = digits.data
    data_train_Y = digits.target
    choose = data_train_Y >= 8
    x = data_train_X[choose].T
    y = data_train_Y[choose] - 8

    n_hidden = 20
    classes = 1
    lr = 1e-3
    threshold = 0.5
    test_index = 5

    xtest = x[:,:test_index]
    ytest = y[:test_index]
    xtrain = x[:,test_index:]
    ytrain = y[test_index:]

    train(xtrain, ytrain, xtest, ytest, n_hidden, classes, lr, threshold, 0)