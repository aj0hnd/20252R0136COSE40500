import os, sys, time, pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tqdm import tqdm
from optimizer import Adam
from chapter_5.layers import *
from torchvision.datasets import MNIST

def convert_to_one_hot(x: np.ndarray, dim: int = 10):
    out = np.zeros((x.shape[0], dim))
    out[np.arange(x.shape[0]), x] = 1.0
    return out

class FiveLayerNet:
    def __init__(self, input_dim=784, hidden_dim=100, output_dim=10, init_type='kaiming', act_type='relu'):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.init_std = 0.01
        self.init_type = init_type
        self.act_type = act_type if init_type != 'xavier' else 'sig'

        self.params = {}
        self.params['W1'] = np.random.normal(0, 1.0, size=(input_dim, hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.normal(0, 1.0, size=(hidden_dim, hidden_dim))
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = np.random.normal(0, 1.0, size=(hidden_dim, hidden_dim))
        self.params['b3'] = np.zeros(hidden_dim)
        self.params['W4'] = np.random.normal(0, 1.0, size=(hidden_dim, hidden_dim))
        self.params['b4'] = np.zeros(hidden_dim)
        self.params['W5'] = np.random.normal(0, 1.0, size=(hidden_dim, output_dim))
        self.params['b5'] = np.zeros(output_dim)
        self.weight_initialize()

        self.layers = {}
        self.layers['Affine1'] = AffineLayer(self.params['W1'], self.params['b1'])
        self.layers['Act1'] = self.activation()
        self.layers['Affine2'] = AffineLayer(self.params['W2'], self.params['b2'])
        self.layers['Act2'] = self.activation()
        self.layers['Affine3'] = AffineLayer(self.params['W3'], self.params['b3'])
        self.layers['Act3'] = self.activation()
        self.layers['Affine4'] = AffineLayer(self.params['W4'], self.params['b4'])
        self.layers['Act4'] = self.activation()
        self.layers['Affine5'] = AffineLayer(self.params['W5'], self.params['b5'])
        self.last_layer = SoftmaxWithLoss()

    def weight_initialize(self):
        if self.init_type == 'xavier':
            for key in ['W1', 'W2', 'W3', 'W4', 'W5']:
                f_in, f_out = self.params[key].shape
                self.params[key] *= np.sqrt(2 / (f_in + f_out))
        if self.init_type == 'kaiming':
            for key in ['W1', 'W2', 'W3', 'W4', 'W5']:
                f_in = self.params[key].shape[0]
                self.params[key] *= (np.sqrt(2 / f_in))
        else:
            for key in ['W1', 'W2', 'W3', 'W4', 'W5']:
                self.params[key] *= self.init_std

    def activation(self):
        if self.act_type == 'sig':
            return SigmoidLayer()
        else:
            return ReLULayer()
    
    def predict(self, x: np.ndarray):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    def get_loss(self, x: np.ndarray, t: np.ndarray):
        predictions = self.predict(x)
        loss = self.last_layer.forward(predictions, t)
        return loss

    def get_accuracy(self, x: np.ndarray, t: np.ndarray):
        predictions = self.predict(x)
        predicted_labels = np.argmax(predictions, axis=1)
        ans_labels = np.argmax(t, axis=1)

        accuracy = np.sum(predicted_labels == ans_labels) / predicted_labels.shape[0]
        return accuracy
    
    def get_gradient(self, x, t):
        self.get_loss(x, t)

        dout = 1
        dout = self.last_layer.backward(dout=dout)
        for layer in reversed(list(self.layers.values())):
            dout = layer.backward(dout)
        
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        grads['W3'], grads['b3'] = self.layers['Affine3'].dW, self.layers['Affine3'].db
        grads['W4'], grads['b4'] = self.layers['Affine4'].dW, self.layers['Affine4'].db
        grads['W5'], grads['b5'] = self.layers['Affine5'].dW, self.layers['Affine5'].db
        return grads
    
    def train(self, x: np.ndarray, t: np.ndarray, optimizer):
        grads = self.get_gradient(x, t)
        self.params = optimizer.update(self.params, grads)

if __name__ == '__main__':
    data_path = '/Users/ungsungko_master/Desktop/self/data'
    result_path = os.path.join(os.getcwd(), 'chapter_6/result')
    weight_path = os.path.join(result_path, 'weight/init')
    figure_path = os.path.join(result_path, 'figure/init')
    os.makedirs(weight_path, exist_ok=True)
    os.makedirs(figure_path, exist_ok=True)

    log = 'train MNIST dataset with relu for different weight initialization.\n\n'
    print('train MNIST dataset with relu for different weight initialization.\n')

    train_loss_dict, train_accur_dict = {}, {}
    test_loss_dict, test_accur_dict = {}, {}
    init_types = ['0.01', 'xavier', 'kaiming']

    hyper_param_set = {
        'epoch': 2000,
        'log_iter': 100,
        'lr': 0.001,
        'beta1': 0.9,
        'beta2': 0.999,
        'batch_size': 1000,
        'act_type': 'relu'
    }

    train_df = MNIST(root=data_path, train=True, download=False)
    test_df = MNIST(root=data_path, train=False, download=False)

    x_train, y_train = train_df.data.reshape(-1, 784) / 255.0, convert_to_one_hot(train_df.targets)
    x_test, y_test = test_df.data.reshape(-1, 784) / 255.0, convert_to_one_hot(test_df.targets)

    
    for init_type in init_types:
        log += f"### MNIST + {hyper_param_set['act_type']} with standard {init_type} ###"
        print(f"### MNIST + {hyper_param_set['act_type']} with standard {init_type} ###")

        pkl_path = os.path.join(weight_path, f"mnist_{hyper_param_set['act_type']}_{init_type}.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as fid:
                network = pickle.load(fid)
        else:
            network = FiveLayerNet(init_type=init_type, act_type=hyper_param_set['act_type'])

        optimizer = Adam(lr=hyper_param_set['lr'], beta1=hyper_param_set['beta1'], beta2=hyper_param_set['beta2'])

        start_time = time.time()
        train_loss_dict[init_type], train_accur_dict[init_type] = [], []
        test_loss_dict[init_type], test_accur_dict[init_type] = [], []
        for epoch in tqdm(range(hyper_param_set['epoch'])):
            batch_mask = np.random.choice(x_train.shape[0], hyper_param_set['batch_size'], replace=False)
            x_batch, y_batch = x_train[batch_mask], y_train[batch_mask]

            network.train(x_batch, y_batch, optimizer=optimizer)

            if epoch % hyper_param_set['log_iter'] == 0:
                loss = network.get_loss(x_batch, y_batch)
                accur = network.get_accuracy(x_batch, y_batch)
                
                train_loss_dict[init_type].append(loss)
                train_accur_dict[init_type].append(accur)

                log += f"\tAccuracy / Loss of train in epoch {epoch+1} / {hyper_param_set['epoch']} with init type {init_type} : {accur:.2f} / {loss:.4f}\n"

                batch_mask = np.random.choice(x_test.shape[0], hyper_param_set['batch_size'], replace=False)
                x_batch, y_batch = x_test[batch_mask], y_test[batch_mask]
                loss = network.get_loss(x_batch, y_batch)
                accur = network.get_accuracy(x_batch, y_batch)

                test_loss_dict[init_type].append(loss)
                test_accur_dict[init_type].append(accur)

                log += f"\tAccuracy / Loss of test in epoch {epoch+1} / {hyper_param_set['epoch']} with init type {init_type} : {accur:.2f} / {loss:.4f}\n"

        with open(pkl_path, 'wb') as fid:
            pickle.dump(network, fid)
        log += f"finished train with init type {init_type}. (spent time: {time.time() - start_time:.4f})\n\n"
        print(f"finished train with init type {init_type}. (spent time: {time.time() - start_time:.4f})\n")
    
    log += 'Train all finished.\n'

    idx = np.arange(0, hyper_param_set['epoch'], hyper_param_set['log_iter'])

    train_path = os.path.join(figure_path, 'train_accur_init.png')
    for init_type in init_types:
        plt.plot(idx, train_accur_dict[init_type], label=f"{init_type}")
    plt.legend()
    plt.xlabel('iter')
    plt.ylabel('accuracy')
    plt.title('Train accuracy per each initial type')
    plt.savefig(train_path)
    plt.close()

    test_path = os.path.join(figure_path, 'test_accur_init.png')
    for init_type in init_types:
        plt.plot(idx, test_accur_dict[init_type], label=f"{init_type}")
    plt.legend()
    plt.xlabel('iter')
    plt.ylabel('accuracy')
    plt.title('Test accuracy per each initial type')
    plt.savefig(test_path)
    plt.close()

    train_path = os.path.join(figure_path, 'train_loss_init.png')
    for init_type in init_types:
        plt.plot(idx, train_loss_dict[init_type], label=f"{init_type}")
    plt.legend()
    plt.xlabel('iter')
    plt.ylabel('loss')
    plt.title('Train loss per each initial type')
    plt.savefig(train_path)
    plt.close()

    test_path = os.path.join(figure_path, 'test_loss_init.png')
    for init_type in init_types:
        plt.plot(idx, test_loss_dict[init_type], label=f"{init_type}")
    plt.legend()
    plt.xlabel('iter')
    plt.ylabel('loss')
    plt.title('Test loss per each initial type')
    plt.savefig(test_path)
    plt.close()

    log_path = os.path.join(result_path, 'log_init.txt')
    with open(log_path, 'w') as fid:
        fid.write(log)