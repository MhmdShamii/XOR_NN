import random

from helpers import sigmoid


class Neuron:
    def __init__(self, num_inputs):
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        self.bias = random.uniform(-1, 1)
        self.activation_value = None

    def activation(self, inputs):
        inputs_and_weights_arr = zip(self.weights,inputs)
        inputs_times_weights_arr = [inVal * weight for inVal, weight in inputs_and_weights_arr ] 
        neuron_value = sum(inputs_times_weights_arr) + self.bias
        self.activation_value = sigmoid(neuron_value)
        return self.activation_value

    def activation_der(self):
        return self.activation_value * (1 - self.activation_value)