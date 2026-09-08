from Layer import Layer
from helpers import blame


class Network:

    _network_output = None

    def __init__(self, input_size:int, layers_sizes:list):
        self.input_layer_size = input_size
        self.num_of_layers = len(layers_sizes)
        self.num_of_neurons_per_layer = layers_sizes

        #start layer creation from raw input layer
        self.network_layers = [Layer(self.num_of_neurons_per_layer[0],self.input_layer_size)]

        for count in range(1, self.num_of_layers):
            self.network_layers.append(Layer(self.num_of_neurons_per_layer[count],self.num_of_neurons_per_layer[count-1]))


    def forward_propagation(self, input_list:list):
        if len(input_list) != self.input_layer_size:
            raise ValueError("Input list size does not match the input layer size.")

        current_outputs = input_list
        for layers in self.network_layers:
            current_outputs = layers.layer_output(current_outputs)

        self._network_output = current_outputs
        return self._network_output

    def back_propagation(self, y_true):
        
        output_blame = blame(self._network_output[0], y_true)

        for layer in self.network_layers[::-1]:
            for neuron in layer.layer_neurons:
                ...

