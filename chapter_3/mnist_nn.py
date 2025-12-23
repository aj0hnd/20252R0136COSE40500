import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, Optional
from torchvision import datasets
from activation import *

def img_show(img, label):
    plt.axis('off')
    plt.imshow(img, cmap='gray')
    plt.title(f"image of {label}")
    plt.show()

class MNISTNN:
    def __init__(self, input_dim: int = 784, hidden_dim: list = [50, 100], output_dim: int = 10, 
                 activation: Literal['step', 'sigmoid', 'relu'] = 'relu', name: str = 'MNIST pipeline'):
        self.name = name

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.activation_type = activation
        self.activation = self.call_func()
        self.softmax = softmax

        self.w1 = np.random.randn(input_dim, hidden_dim[0]).astype(np.float32)
        self.w2 = np.random.randn(hidden_dim[0], hidden_dim[1]).astype(np.float32)
        self.w3 = np.random.randn(hidden_dim[1], output_dim).astype(np.float32)

        self.b1 = np.zeros(hidden_dim[0])
        self.b2 = np.zeros(hidden_dim[1])
        self.b3 = np.zeros(output_dim)
    
    def call_func(self):
        if self.activation_type == 'step':
            return step
        if self.activation_type == 'sig':
            return sigmoid
        if self.activation_type == 'relu':
            return relu

    def forward(self, inputs):
        a1 = np.dot(inputs, self.w1) + self.b1[None, :]
        h1 = self.activation(a1)

        a2 = np.dot(h1, self.w2) + self.b2[None, :]
        h2 = self.activation(a2)

        a3 = np.dot(h2, self.w3) + self.b3[None, :]
        out = self.softmax(a3)
        return out

if __name__ == '__main__':
    data_path = '/Users/ungsungko_master/Desktop/self/data'

    train_df = datasets.MNIST(root=data_path, download=True, train=True)
    x_train, y_train = train_df.data.numpy(), train_df.targets.numpy()
    test_df = datasets.MNIST(root=data_path, download=True, train=False)
    x_test, y_test = test_df.data.numpy(), test_df.targets.numpy()

    print(f"MNIST train image len: {len(x_train)}")
    print(f"MNIST train image shape: {x_train.shape}")
    print(f"MNIST train label shape: {y_train.shape}")
    print(f"MNIST test image len: {len(x_test)}")
    print(f"MNIST test image shape: {x_test.shape}")
    print(f"MNIST test label shape: {y_test.shape}")

    # img_show(x_train[0], y_train[0])

    mnist_network = MNISTNN()
    outputs = mnist_network.forward(inputs=x_test.reshape(-1, 784))
    print(f"shape of ouputs: {outputs.shape}")