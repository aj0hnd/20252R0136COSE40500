import numpy as np

def softmax(x: np.ndarray):
    origin_shape = x.shape
    if x.ndim == 1:
        x = x.reshape(1, -1)
    x = x - np.max(x, axis=1, keepdims=True)
    y = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    if len(origin_shape) == 1:
        y = y.reshape(-1)
    return y

def cross_entropy_loss(y: np.ndarray, t: np.ndarray):
    if len(y.shape) == 1:
        y = y.reshape(1, -1)
        t = t.reshape(1, -1)
    return -np.sum(t * np.log(y + 1e-5)) / y.shape[0]

class ReLULayer:
    def __init__(self):
        pass
    
    def forward(self, x):
        self.x = x
        self.mask = x < 0

        out = x.copy()
        out[self.mask] = 0.
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0.
        return dout
    
class SigmoidLayer:
    def __init__(self):
        pass

    def forward(self, x):
        self.x = x
        self.out = 1 / (1 + np.exp(-x))
        return self.out
    
    def backward(self, dout):
        dx = dout * self.out * (1 - self.out)
        return dx
    
class AffineLayer:
    def __init__(self, W, b):
        self.W = W
        self.b = b
    
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx  
    
class SoftmaxWithLoss:
    def __init__(self):
        self.t = None
        self.y = None
        self.loss = None
    
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_loss(self.y, self.t)
        return self.loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0] if len(self.t.shape) > 1 else 1
        dx = (self.y - self.t) / batch_size
        return dx