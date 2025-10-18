import neuralnetworkdev as nndev
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0.5, 0.7, 1, 1, 0.3]])
Y = np.array([[0, 1]])

nn = nndev.FullyConnected1DModel(input_size=5,topology=[2],act_f="relu")

print(nn.apply(X))

losses = nn.compile(X, Y, 'MSE',50)

print(nn.layers[0].act_f(X))
print(nn.apply(X))

plt.plot(range(len(losses)), losses)
plt.show()

