import numpy as np

class SGD:
    def __init__(self, lr=0.001):
        self.lr = lr
    
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]
        return params

class Momentum:
    def __init__(self, lr=0.001, alpha=0.9):
        self.lr = lr
        self.alpha = alpha
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.alpha * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]
        return params

class AdaGrad:
    def __init__(self, lr=0.01):
        self.eps = 1e-7
        self.lr = lr
        self.h = None
    
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        
        for key in params.keys():
            self.h[key] += grads[key] ** 2
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + self.eps)
        return params

class RMSProp:
    def __init__(self, lr=0.001, p=0.999):
        self.eps = 1e-7
        self.p = p
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        
        for key in params.keys():
            self.h[key] = self.p * (self.h[key]) + (1 - self.p) * (grads[key] * grads[key])
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + self.eps)
        return params

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.eps = 1e-7
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.m1 = None # for momentum logic
        self.m2 = None # for RMSProp logic

    def update(self, params, grads, iter=None):
        if self.m1 is None:
            self.m1, self.m2 = {}, {}
            for key, val in params.items():
                self.m1[key] = np.zeros_like(val)
                self.m2[key] = np.zeros_like(val)
        
        for key in params.keys():
            self.m1[key] = self.beta1 * self.m1[key] + (1 - self.beta1) * grads[key]
            self.m2[key] = self.beta2 * self.m2[key] + (1 - self.beta2) * grads[key] * grads[key]

            
            if iter is not None:
                m1 = self.m1[key] / (1 - self.beta1 ** iter)
                m2 = self.m2[key] / (1 - self.beta2 ** iter)
                params[key] -= self.lr * m1 / (np.sqrt(m2) + self.eps)
            else:
                params[key] -= self.lr * self.m1[key] / (np.sqrt(self.m2[key]) + self.eps)
        return params
    
            

