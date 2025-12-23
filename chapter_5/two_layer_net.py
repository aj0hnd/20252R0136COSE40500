import numpy as np
from layers import *
from collections import OrderedDict

class TwoLayerNet:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, weight_init_std: float = 0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.params = {}
        self.params['W1'] = np.random.normal(loc=0.0, scale=weight_init_std, size=(input_dim, hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.normal(loc=0.0, scale=weight_init_std, size=(hidden_dim, output_dim))
        self.params['b2'] = np.zeros(output_dim)
        self.param_list = ['W1', 'b1', 'W2', 'b2']

        self.layers = OrderedDict()
        self.layers['Affine1'] = AffineLayer(W=self.params['W1'], b=self.params['b1'])
        self.layers['Relu1'] = ReLULayer()
        self.layers['Affine2'] = AffineLayer(W=self.params['W2'], b=self.params['b2'])
        
        self.last_layer = SoftmaxWithLoss()
    
    def predict(self, x: np.ndarray):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def get_loss(self, x: np.ndarray, t: np.ndarray):
        predictions = self.predict(x)
        loss = self.last_layer.forward(predictions, t)
        return loss
    
    def get_accuracy(self, x: np.ndarray, t: np.ndarray):
        if x.ndim == 1:
            x = x.reshape(1, -1)
            t = t.reshape(1, -1)
        predictions = self.predict(x)
        predicted_labels = np.argmax(predictions, axis=1)
        correct_labels = np.argmax(t, axis=1)
        accuracy = np.sum(predicted_labels == correct_labels) / t.shape[0]
        return accuracy
    
    def get_gradient(self, x: np.ndarray, t: np.ndarray):
        self.get_loss(x, t)

        dout = 1
        dout = self.last_layer.backward(dout=dout)

        for layer in reversed(list(self.layers.values())):
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        return grads
    
    def train(self, x: np.ndarray, t: np.ndarray, lr: float = 0.1):
        grads = self.get_gradient(x, t)
        for p in self.param_list:
            self.params[p] -= lr * grads[p]
        
