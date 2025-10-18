import neuralnetworkdev as nndev
import numpy as np
import matplotlib.pyplot as plt

training_size = 200

inputs = []
expected = []

radius = 0.1

for _ in range(training_size):
    coords = np.random.rand(2)*0.5+0.25
    circle_cords = 0
    if np.sum((coords-0.5)**2)< radius**2:
        circle_cords = 1
    inputs.append(coords)
    expected.append([circle_cords])

inputs = np.array(inputs)
expected = np.array(expected)

plt.scatter(inputs[expected[:,0] == 0, 0], inputs[expected[:,0] == 0, 1], c="skyblue")
plt.scatter(inputs[expected[:,0] == 1, 0], inputs[expected[:,0] == 1, 1], c="salmon")
plt.axis([0,1,0,1])
plt.show()

nn = nndev.FullyConnected1DModel(input_size=2,topology=[4,8,1],act_f="sigm")

losses = nn.compile(inputs, expected, 'MSE', steps=1000, learning_rate=0.1, show_on_step=10)

def show_output():
    res = 100
    x = np.linspace(-0.1, 1.1, res)
    y = np.linspace(-0.1, 1.1, res)
    outputs = np.zeros((res, res))

    for i0,x0 in enumerate(x):
        for i1,x1 in enumerate(y):
            outputs[i0,i1] = nn.apply(np.array([x0,x1]))[0,0]
        
    fig,(ax1,ax2) = plt.subplots(1,2)
    ax1.pcolormesh(x,y,outputs,cmap = "coolwarm")
    ax1.scatter(inputs[expected[:,0] == 0, 0], inputs[expected[:,0] == 0, 1], c="skyblue")
    ax1.scatter(inputs[expected[:,0] == 1, 0], inputs[expected[:,0] == 1, 1], c="salmon")
    ax2.plot(range(len(losses)), losses)
    plt.show()

show_output()