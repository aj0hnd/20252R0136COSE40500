import numpy as np

class BatchNorm:
    def __init__(self, gamma, delta):
        self.eps = 1e-7
        self.gamma = gamma # (d,)
        self.delta = delta # (d,)

        self.momentum = 0.9
        self.moving_mean = np.zeros_like(gamma)
        self.moving_var = np.zeros_like(gamma)

    def forward(self, x: np.ndarray, training_flg=True): # (N,d)
        if training_flg:
            self.x = x
            self.mean = np.mean(x, axis=0, keepdims=True) # (1,d)
            self.var = np.mean((x - self.mean) ** 2, axis=0, keepdims=True) # (1,d)
            self.xmu = self.x - self.mean # (N,d)
            self.rvar = np.sqrt(self.var + self.eps) # (1,d)
            self.ivar = 1 / self.rvar # (1,d)
            self.x_hat = self.xmu * self.ivar # (N,d)
            out = self.gamma * self.x_hat + self.delta # (N,d)

            self.moving_mean = self.momentum * self.moving_mean + (1 - self.momentum) * self.mean
            self.moving_var = self.momentum * self.moving_var + (1 - self.momentum) * self.var
        else:
            x_hat = (x - self.moving_mean) / np.sqrt(self.moving_var + self.eps)
            out = self.gamma * x_hat + self.delta

        return out
    
    def backward(self, dout: np.ndarray): # (N,d)
        batch_size = self.x.shape[0]
        self.dgamma = np.sum(dout * self.x_hat, axis=0, keepdims=True) # (1,d)
        self.ddelta = np.sum(dout, axis=0, keepdims=True) # (1,d)

        dx_hat = dout * self.gamma # (N,d)

        divar = np.sum(self.xmu * dx_hat, axis=0, keepdims=True) # (1, d)
        drvar = -1 * self.ivar ** 2 * divar # (1,d)
        dvar = 0.5 * self.ivar * drvar # (1,d)

        dmx2 = np.ones(self.x.shape) * dvar / batch_size # (N,d)
        dxmu = self.ivar * dx_hat + 2 * self.xmu * dmx2 # (N,d)

        dmu = -np.sum(dxmu, axis=0, keepdims=True)
        dmx = np.ones(self.x.shape) * dmu / batch_size
        self.dx = dmx + dxmu
        return self.dx

    def overload(self, gamma, delta):
        self.gamma = gamma
        self.delta = delta

class DropOut:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)
    
    def backward(self, dout):
        return dout * self.mask

if __name__ == '__main__':
    x = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])
    print(np.mean(x, axis=1))