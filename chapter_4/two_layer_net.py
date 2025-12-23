import os, sys
import numpy as np
from typing import Literal
from loss import get_ce
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))
from chapter_3.activation import *

class TwoLayerNet:
    def __init__(self, input_dim: int = 784, hidden_dim: int = 100, output_dim: int = 10, 
                 activation_type: Literal['sig', 'relu'] = 'sig', weight_init_std: float = 0.01):
        self.weight_types = ['W1', 'b1', 'W2', 'b2']
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        if activation_type == 'sig': self.activation = sigmoid
        else: self.activation = relu

        self.eps = 1e-4
        self.softmax = softmax
        self.loss_func = get_ce

        self.params = {}
        self.params['W1'] = np.random.normal(loc=0.0, scale=weight_init_std, size=(input_dim, hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.normal(loc=0.0, scale=weight_init_std, size=(hidden_dim, output_dim))
        self.params['b2'] = np.zeros(output_dim)

    def forward(self, inputs: np.ndarray):
        a1 = inputs.dot(self.params['W1']) + self.params['b1']
        h1 = self.activation(a1)

        a2 = h1.dot(self.params['W2']) + self.params['b2']
        out = self.softmax(a2)
        return out
    
    def get_accuracy(self, inputs: np.ndarray, labels: np.ndarray):
        predictions = self.forward(inputs)
        predicted_labels = np.argmax(predictions, axis=1)
        answers = np.argmax(labels, axis=1)

        accuracy = np.sum(predicted_labels == answers) / predictions.shape[0]
        return accuracy
    
    def get_loss(self, inputs: np.ndarray, labels: np.ndarray):
        predictions = self.forward(inputs)
        loss = self.loss_func(predictions, labels)
        return loss
    
    def numerical_gradient_per_param(self, func, param_type):
        grad = np.zeros_like(self.params[param_type])
        it = np.nditer(self.params[param_type], flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            val = self.params[param_type][idx]
            
            self.params[param_type][idx] = val + self.eps
            f_right = func(0)
            self.params[param_type][idx] = val - self.eps
            f_left = func(0)

            grad[idx] = (f_right - f_left) / (2 * self.eps)
            it.iternext()
        return grad
    
    def numerical_gradients(self, inputs: np.ndarray, labels: np.ndarray):
        temp_func = lambda _: self.get_loss(inputs, labels)

        grads = {}
        grads['W1'] = self.numerical_gradient_per_param(func=temp_func, param_type='W1')
        grads['b1'] = self.numerical_gradient_per_param(func=temp_func, param_type='b1')
        grads['W2'] = self.numerical_gradient_per_param(func=temp_func, param_type='W2')
        grads['b2'] = self.numerical_gradient_per_param(func=temp_func, param_type='b2')

        return grads
    
    def train(self, inputs: np.ndarray, labels: np.ndarray, lr: float = 0.01):
        grads = self.numerical_gradients(inputs=inputs, labels=labels)
        for weight in self.weight_types:
            self.params[weight] -= lr * grads[weight]
        

if __name__ == '__main__':
    network = TwoLayerNet()

    random_x = np.random.randn(3, 784)
    random_y, y_mask = np.zeros((3, 10)), np.random.randint(low=0, high=9, size=3)
    random_y[np.arange(3), y_mask.flatten()] = 1.

    print(f"Shape of forward batch 3: {network.forward(random_x).shape}")
    print(f"Value of loss of batch 3: {network.get_loss(random_x, random_y):.4f}")
    
    grads = network.numerical_gradients(random_x, random_y)
    for weight_t in network.weight_types:
        print(f"Shape of gradient of {weight_t}: {grads[weight_t].shape}")
    
    try:
        network.train(random_x, random_y)
        print("\nTrain succeed.")
    except:
        print("\nTrain failed.")

    print(f"Get initial accuracy: {network.get_accuracy(random_x, random_y)}")
