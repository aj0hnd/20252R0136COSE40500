import os, sys, time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from options import BatchNorm
from optimizer import Adam
from torchvision.datasets import MNIST

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))
from chapter_5.layers import *

def convert_to_one_hot(x: np.ndarray):
    out = np.zeros((x.shape[0], 10))
    out[np.arange(x.shape[0]), x] = 1.0
    return out


class MLP:
    def __init__(self, input_dim=784, hidden_dim=100, output_dim=10, num_layer=5, init_type='xaiming', use_bn = True):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.use_bn = use_bn
        self.init_type = init_type
        self.num_layer = num_layer
        self.layer_dim = [input_dim] + [hidden_dim] * (num_layer-1) + [output_dim]

        self.params = {}
        for idx in range(num_layer):
            w_name, b_name = 'W' + str(idx+1), 'b' + str(idx+1)
            g_name, d_name = 'd' + str(idx+1), 'g' + str(idx+1)
            self.params[w_name] = np.random.normal(0, 1.0, size=(self.layer_dim[idx], self.layer_dim[idx+1]))
            self.params[b_name] = np.zeros(self.layer_dim[idx+1])
            if self.use_bn:
                self.params[g_name] = np.ones((1, self.layer_dim[idx+1]))
                self.params[d_name] = np.zeros((1, self.layer_dim[idx+1]))
        self.weight_initialize()

        self.layers = {}
        for idx in range(num_layer):
            w_name, b_name = 'W' + str(idx+1), 'b' + str(idx+1)
            g_name, d_name = 'd' + str(idx+1), 'g' + str(idx+1)
            layer_name, bn_name, act_name = 'Affine' + str(idx+1), 'BatchNorm' + str(idx+1), 'Act' + str(idx+1)
            self.layers[layer_name] = AffineLayer(self.params[w_name], self.params[b_name])
            if self.use_bn:
                self.layers[bn_name] = BatchNorm(self.params[g_name], self.params[d_name], self.layer_dim[idx+1])
            if idx < num_layer - 1:
                self.layers[act_name] = ReLULayer()
        self.last_layer = SoftmaxWithLoss()

    def weight_initialize(self):
        for idx in range(self.num_layer):
            w_name = 'W' + str(idx+1)
            if self.init_type == 'kaiming':
                self.params[w_name] *= np.sqrt(2 / self.layer_dim[idx])
            if self.init_type == 'xavier':
                self.params[w_name] *= np.sqrt(2 / (self.layer_dim[idx] + self.layer_dim[idx+1]))
            if isinstance(self.init_type, (int, float)):
                self.params[w_name] += self.init_type

    def predict(self, x: np.ndarray):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    def get_loss(self, x: np.ndarray, t: np.ndarray):
        predictions = self.predict(x)
        loss = self.last_layer.forward(predictions, t)
        return loss
    
    def get_accuracy(self, x: np.ndarray, t: np.ndarray):
        batch_size = x.shape[0]
        predictions = self.predict(x)
        predicted_labels = np.argmax(predictions, axis=1)
        ans_labels = np.argmax(t, axis=1)

        accuracy = np.sum(predicted_labels == ans_labels) / batch_size
        return accuracy
    
    def get_gradient(self, x: np.ndarray, t: np.ndarray):
        self.get_loss(x, t)

        dout = 1
        dout = self.last_layer.backward(dout)
        for layer in reversed(list(self.layers.values())):
            dout = layer.backward(dout)
        
        grads = {}
        for idx in range(self.num_layer):
            w_name, b_name = 'W' + str(idx+1), 'b' + str(idx+1)
            g_name, d_name = 'd' + str(idx+1), 'g' + str(idx+1)
            layer_name, bn_name = 'Affine' + str(idx+1), 'BatchNorm' + str(idx+1)

            grads[w_name], grads[b_name] = self.layers[layer_name].dW, self.layers[layer_name].db
            if self.use_bn:
                grads[g_name], grads[d_name] = self.layers[bn_name].dgamma, self.layers[bn_name].ddelta
        return grads
    
    def train(self, x: np.ndarray, t: np.ndarray, optimizer):
        grads = self.get_gradient(x, t)
        self.params = optimizer.update(self.params, grads)

if __name__ == '__main__':
    data_path = '/Users/ungsungko_master/Desktop/self/data'
    result_path = os.path.join(os.getcwd(), 'chapter_6/result')
    weight_path = os.path.join(result_path, 'weight/batchnorm')
    figure_path = os.path.join(result_path, 'figure/batchnorm')
    os.makedirs(weight_path, exist_ok=True)
    os.makedirs(figure_path, exist_ok=True)

    log = 'Train MNIST dataset compare w w/o batch normalization.\n'

    train_loss_dict, test_loss_dict = {}, {}
    train_accur_dict, test_accur_dict = {}, {}

    train_df = MNIST(root=data_path, train=True, download=False)
    test_df = MNIST(root=data_path, train=False, download=False)

    hyper_param_set = {
        'num_layer': 2,
        'epoch': 2000,
        'log_iter': 100,
        'lr': 0.001,
        'beta1': 0.9,
        'beta2': 0.999,
        'batch_size': 1000,
        'init_type': 'kaiming'
    }

    x_train, y_train = train_df.data.reshape(-1, 784) / 255.0, convert_to_one_hot(train_df.targets)
    x_test, y_test = test_df.data.reshape(-1, 784) / 255.0, convert_to_one_hot(test_df.targets)

    for use_bn in [True, False]:

        # if not use_bn:
        #     hyper_param_set['epoch'] = 20000
        #     hyper_param_set['log_iter'] = 1000
        #     hyper_param_set['lr'] = 0.01
        
        log += f"### start trainig MNIST using batchnorm ({use_bn}) ###\n"

        pkl_path = os.path.join(weight_path, f"mnist_with_bn_{use_bn}_{hyper_param_set['num_layer']}.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as fid:
                network = pickle.load(fid)
        else:
            network = MLP(use_bn=use_bn, num_layer=hyper_param_set['num_layer'])

        optimizer = Adam(hyper_param_set['lr'], hyper_param_set['beta1'], hyper_param_set['beta2'])

        idx = 1 if use_bn else 0
        train_loss_dict[idx], test_loss_dict[idx] = [], []
        train_accur_dict[idx], test_accur_dict[idx] = [], []

        start_time = time.time()
        for epoch in tqdm(range(hyper_param_set['epoch'])):
            batch_mask = np.random.choice(x_train.shape[0], hyper_param_set['batch_size'], replace=False)
            x_batch, y_batch = x_train[batch_mask], y_train[batch_mask]

            network.train(x_batch, y_batch, optimizer)

            if epoch % hyper_param_set['log_iter'] == 0:
                loss, accur = network.get_loss(x_batch, y_batch), network.get_accuracy(x_batch, y_batch)
                train_loss_dict[idx].append(loss)
                train_accur_dict[idx].append(accur)
                log += f"\tAccuracy / Loss of train in epoch {epoch+1} / {hyper_param_set['epoch']} with batch norm ({use_bn}) : {accur:.2f} / {loss:.4f}\n"

                batch_mask = np.random.choice(x_test.shape[0], hyper_param_set['batch_size'], replace=False)
                x_batch, y_batch = x_test[batch_mask], y_test[batch_mask]

                loss, accur = network.get_loss(x_batch, y_batch), network.get_accuracy(x_batch, y_batch)
                test_loss_dict[idx].append(loss)
                test_accur_dict[idx].append(accur)
                log += f"\tAccuracy / Loss of test in epoch {epoch+1} / {hyper_param_set['epoch']} with batch norm ({use_bn}) : {accur:.2f} / {loss:.4f}\n"

        with open(pkl_path, 'wb') as fid:
            pickle.dump(network, fid)
        log += f"finished train with batch norm ({use_bn}). (spent time: {time.time() - start_time:.4f})\n\n"

    log += 'Train all finished.\n'

    idx = np.arange(0, hyper_param_set['epoch'], hyper_param_set['log_iter'])

    train_loss_path = os.path.join(figure_path, f"train_loss_batch_norm_{hyper_param_set['num_layer']}.png")
    train_accur_path = os.path.join(figure_path, f"train_accur_batch_norm_{hyper_param_set['num_layer']}.png")
    test_loss_path = os.path.join(figure_path, f"test_loss_batch_norm_{hyper_param_set['num_layer']}.png")
    test_accur_path = os.path.join(figure_path, f"test_accur_batch_norm_{hyper_param_set['num_layer']}.png")

    plt.plot(idx, train_loss_dict[1], color='blue', label='w bn')
    plt.plot(idx, train_loss_dict[0], color='r', label='w/o bn')
    plt.legend()
    plt.xlabel('iter')
    plt.ylabel('loss')
    plt.title("Compare train loss between w and w/o batch normalization.")
    plt.savefig(train_loss_path)
    plt.close()

    plt.plot(idx, test_loss_dict[1], color='blue', label='w bn')
    plt.plot(idx, test_loss_dict[0], color='r', label='w/o bn')
    plt.legend()
    plt.xlabel('iter')
    plt.ylabel('loss')
    plt.title("Compare test loss between w and w/o batch normalization.")
    plt.savefig(test_loss_path)
    plt.close()

    plt.plot(idx, train_accur_dict[1], color='blue', label='w bn')
    plt.plot(idx, train_accur_dict[0], color='r', label='w/o bn')
    plt.legend()
    plt.xlabel('iter')
    plt.ylabel('accuracy')
    plt.title("Compare train accuracy between w and w/o batch normalization.")
    plt.savefig(train_accur_path)
    plt.close()

    plt.plot(idx, test_accur_dict[1], color='blue', label='w bn')
    plt.plot(idx, test_accur_dict[0], color='r', label='w/o bn')
    plt.legend()
    plt.xlabel('iter')
    plt.ylabel('accuracy')
    plt.title("Compare test accuracy between w and w/o batch normalization.")
    plt.savefig(test_accur_path)
    plt.close()    

    log_path = os.path.join(result_path, 'log_bn.txt')
    with open(log_path, 'w') as fid:
        fid.write(log)