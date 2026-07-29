import random

from helpers import sigmoid


class Neuron:
    def __init__(self, num_inputs):
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        self.bias = random.uniform(-1, 1)

    def activation(self, inputs):
        inputs_and_weights_arr = zip(self.weights,inputs)
        inputs_times_weights_arr = [inVal * weight for inVal, weight in inputs_and_weights_arr ] 
        neuron_value = sum(inputs_times_weights_arr) + self.bias
        return sigmoid(neuron_value)