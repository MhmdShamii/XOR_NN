from Network import Network
from helpers import loss_MSE
input_data = [1, 0]
target_output = [0]


net = Network(2, [4, 1])
output = net.forward_propagation(input_data)
print("Output of the network:", output)
print("Mean Squared Error:", loss_MSE(target_output[0], output[0]))