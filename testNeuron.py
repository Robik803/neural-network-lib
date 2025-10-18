import neuralnetworkdev as nndev
import numpy as np

def f(x):
    return x

a = np.array([4,5,-6,3,5])
b = 5
c = np.random.rand(5)


n = nndev.Neuron(5,"sigm")

print(n.apply(a,returnWeightedSum=True))

print(c.shape)

