import numpy as np
import matplotlib.pyplot as plt
from typing import Union

def sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))

def relu(x: np.ndarray):
    out = x.copy()
    out[x < 0] = 0.0
    return out

def sigmoid_weight_init(std: Union[float, str]):
    node_num = 100
    num_hidden_layer = 5

    activations = {}
    x = np.random.randn(1000, node_num)
    for i in range(num_hidden_layer):
        if i != 0: x = activations[i-1]

        if isinstance(std, float):
            weight = np.random.normal(loc=0.0, scale=std, size=(node_num, node_num))
        else:
            weight = np.random.normal(loc=0.0, scale=1 / np.sqrt(node_num), size=(node_num, node_num))
        a = np.dot(x, weight)
        h = sigmoid(a)
        activations[i] = h
    
    fig, ax = plt.subplots(1, num_hidden_layer, figsize=(15, 4))
    plt.subplots_adjust(wspace=1.0)
    for idx, act in activations.items():
        ax[idx].set_title(f"{idx+1}-layer")
        ax[idx].hist(act.flatten(), 30, range=(0, 1))
        ax[idx].set_ylim(0, 7000)
    plt.suptitle(f"sigmoid activation with init std {std}")
    plt.show()
    plt.close()

def relu_weight_init(std: Union[float, str]):
    node_num = 100
    num_hidden_layer = 5

    activations = {}
    x = np.random.randn(1000, node_num)
    for i in range(num_hidden_layer):
        if i != 0: x = activations[i-1]
        
        if isinstance(std, float):
            weight = np.random.normal(loc=0.0, scale=std, size=(node_num, node_num))
        elif std == 'xavier':
            weight = np.random.normal(loc=0.0, scale=np.sqrt(1 / node_num), size=(node_num, node_num))
        else:
            weight = np.random.normal(loc=0.0, scale=np.sqrt(2 / node_num), size=(node_num, node_num))

        a = np.dot(x, weight)
        h = relu(a)
        activations[i] = h
    
    fig, ax = plt.subplots(1, num_hidden_layer, figsize=(15, 4))
    plt.subplots_adjust(wspace=1.0, )
    for idx, act in activations.items():
        ax[idx].set_title(f"{idx+1}-layer")
        ax[idx].hist(act.flatten(), 30, range=(0, 1))
        ax[idx].set_ylim(0, 7000)
    plt.suptitle(f"relu activation with init std {std}")
    plt.show()
    plt.close()

if __name__ == '__main__':
    sigmoid_weight_init(std=1.0)
    sigmoid_weight_init(std=0.01)
    sigmoid_weight_init(std='xavier')

    relu_weight_init(std=0.01)
    relu_weight_init(std='xavier')
    relu_weight_init(std='he')