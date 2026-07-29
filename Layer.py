from Neuron import Neuron

class Layer: 
    def __init__(self, num_of_neurons, inputs_per_neuron):
        self.num_of_neurons = num_of_neurons
        self.inputs_per_neuron = inputs_per_neuron
        self.layer_neurons = [Neuron(self.inputs_per_neuron) for _ in range(self.num_of_neurons)]

    def layer_output(self,inputs_list):
        return [neuron.activation(inputs_list) for neuron in self.layer_neurons]
