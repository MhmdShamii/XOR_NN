from Network import Network
from helpers import loss_MSE,blame

input_data = [[1, 1],[1, 0],[0, 1],[0, 0]]
target_output = [1, 0, 0, 1]


net = Network(2, [4, 1])
output = net.forward_propagation(input_data[0])
print("Output of the network:", output)
print("Mean Squared Error:", loss_MSE(target_output[0], output[0]))
print("Blame:", blame(output[0], target_output[0]))