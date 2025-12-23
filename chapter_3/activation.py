import numpy as np
import matplotlib.pyplot as plt
from typing import Literal

def step(inputs: np.ndarray):
    return (inputs > 0).astype(np.float32)

def sigmoid(inputs: np.ndarray):
    return 1 / (1 + np.exp(-inputs))

def relu(inputs: np.ndarray):
    outputs = np.copy(inputs)
    outputs[inputs < 0] = 0.
    return outputs

def softmax(inputs: np.ndarray):
    if len(inputs.shape) == 1:
        inputs.reshape(1, -1)
    outputs = np.copy(inputs)
    max_ele = np.max(outputs, axis=1, keepdims=True)
    sum_exp = np.sum(np.exp(outputs - max_ele), axis=1, keepdims=True)
    return np.exp(outputs - max_ele) / (sum_exp + 1e-5)

def visualize(func: Literal['step', 'sigmoid', 'relu'] = 'step'):
    x = np.arange(start=-5, stop=5, step=0.05)
    
    if func == 'step':
        y = step(x)
    
    if func == 'sigmoid':
        y = sigmoid(x)

    if func == 'relu':
        y = relu(x)

    plt.grid()
    plt.plot(x, y, color='b')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'{func} function')
    plt.show()
    return


if __name__ == '__main__':
    visualize(func='step')