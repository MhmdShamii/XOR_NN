import math

def sigmoid(neuVal):
    return 1 / (1 + math.exp(-neuVal)) 

def loss_MSE(y_true, y_pred):
    return (y_true - y_pred) ** 2