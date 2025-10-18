import neuralnetworkdev as nndev
import numpy as np

layer1 = nndev.FullyConnectedLayer(n_layer=10,input_size=15,act_f="sigm")

input1 = np.random.rand(15)
input2 = np.random.rand(15)
inputs = np.array([input1,input2])

print(layer1.apply(inputs))



#print(layer1.apply(inputs))