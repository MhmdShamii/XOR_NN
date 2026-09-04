import math

def sigmoid(neuVal):
    return 1 / (1 + math.exp(-neuVal)) 

def loss_MSE(y_true, y_pred):
    return (y_true - y_pred) ** 2

def blame(y_pred, y_true):
    return 2 * (y_pred - y_true)