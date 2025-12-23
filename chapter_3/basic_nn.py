import numpy as np
from typing import Literal
from activation import *
class BasicNN:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 activation: Literal['step', 'sig', 'relu'] = 'relu', name: str = '3_dim_nn'):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.name = name
        self.activation = self.call_func(func=activation)

        self.w1 = np.ones((hidden_dim, input_dim), dtype=np.float32)
        self.w2 = np.ones((output_dim, hidden_dim), dtype=np.float32)
        
        self.b1 = np.zeros((hidden_dim), dtype=np.float32)
        self.b2 = np.zeros((output_dim), dtype=np.float32)
    
    def __str__(self):
        info = ''
        info += f"input dim: {self.input_dim} / hidden dim: {self.hidden_dim} / output dim: {self.output_dim}"
        return info
    
    def call_func(self, func: Literal['step', 'sig', 'relu'] = 'relu'):
        if func == 'step':
            return step
        if func == 'sig':
            return sigmoid
        if func == 'relu':
            return relu
    
    def forward(self, inputs: np.ndarray):
        a1 = np.dot(inputs, self.w1.T) + self.b1
        h1 = self.activation(a1)

        a2 = np.dot(h1, self.w2.T)
        out = self.activation(a2) + self.b2
        return out
    
if __name__ == '__main__':
    my_nn = BasicNN(input_dim=10, hidden_dim=20, output_dim=2)

    inputs = np.random.rand(3, 10)
    outputs = my_nn.forward(inputs=inputs)
    print(f"shape of output: {outputs.shape}")