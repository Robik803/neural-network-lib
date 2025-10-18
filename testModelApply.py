import neuralnetworkdev as nndev
import numpy as np

input1 = np.random.rand(15)
input2 = np.random.rand(15)
inputs = np.array([input1,input2])

layer1 = nndev.FullyConnectedLayer(n_layer=10,input_size=15,act_f="sigm")
layer2 = nndev.FullyConnectedLayer(n_layer=5,input_size=10,act_f="sigm")
layer3 = nndev.FullyConnectedLayer(n_layer=2,input_size=5,act_f="sigm")
layer4 = nndev.FullyConnectedLayer(n_layer=1,input_size=2,act_f="sigm")

nn = nndev.FullyConnected1DModel()
nn.add(layer1)
nn.add(layer2)
nn.add(layer3)
nn.add(layer4)


output = nn.apply(inputs)
print(output)

expected = np.array([[0.78877789], [0.79023663]])

print(nndev.loss('MSE', expected, output))