import numpy as np
import matplotlib.pyplot as plt

def plot_xor_gate():
    # xor gate 0
    coords_0 = np.array([
        [0, 0],
        [1, 1]
    ])
    # xor gate 1
    coords_1 = np.array([
        [0, 1],
        [1, 0]
    ])
    # threshold line
    x = np.arange(start=-0.5, stop=1.5, step=0.01)
    y = -x + 0.5

    plt.grid()
    plt.title('XOR Gate')
    plt.scatter(coords_0[:, 0], coords_0[:, 1], color='r')
    plt.scatter(coords_1[:, 0], coords_1[:, 1], color='b')
    plt.plot(x, y, color='black', label='threshold')
    plt.legend()
    plt.show()

plot_xor_gate()