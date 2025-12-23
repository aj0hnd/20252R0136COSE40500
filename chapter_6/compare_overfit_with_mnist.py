import os, sys, time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import OrderedDict
from torchvision.datasets import MNIST
from options import BatchNorm, DropOut
from optimizer import Adam

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))
from chapter_5.layers import *

def sign(x: np.ndarray):
    out = np.zeros_like(x)
    mask = x > 0
    out[mask] = 1.0
    mask = x < 0
    out[mask] = -1.0
    return out

def convert_to_one_hot(x: np.ndarray):
    out = np.zeros((x.shape[0], 10))
    out[np.arange(x.shape[0]), x] = 1.0
    return out

class MLP:
    def __init__(self, input_dim=784, hidden_dim=100, output_dim=10, 
                 num_layer=10, init_type='kaiming', decay_type=None, decay_ratio=0.001, use_dropout=True, dropout_ratio=0.5):

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.num_layer = num_layer
        self.init_type = init_type
        self.decay_type = decay_type
        self.decay_ratio = decay_ratio
        self.use_dropout = use_dropout
        self.dropout_ratio = dropout_ratio
        self.train_flg = True

        self.layer_dim = [input_dim] + [hidden_dim] * (num_layer - 1) + [output_dim]

        self.params = OrderedDict()
        for idx in range(num_layer):
            w_name, b_name = 'W' + str(idx+1), 'b' + str(idx+1)
            g_name, d_name = 'gamma' + str(idx+1), 'delta' + str(idx+1)
            self.params[w_name] = np.random.normal(0, 1.0, size=(self.layer_dim[idx], self.layer_dim[idx+1]))
            self.params[b_name] = np.zeros(self.layer_dim[idx+1])
            if idx < num_layer - 1:
                self.params[g_name] = np.ones((1, self.layer_dim[idx+1]))
                self.params[d_name] = np.zeros((1, self.layer_dim[idx+1]))
        self.weight_initialize()

        self.layers = {}
        for idx in range(num_layer):
            w_name, b_name = 'W' + str(idx+1), 'b' + str(idx+1)
            g_name, d_name = 'gamma' + str(idx+1), 'delta' + str(idx+1)

            layer_name, act_name = 'Affine' + str(idx+1), 'relu' + str(idx+1)
            bn_name, dropout_name = 'bn' + str(idx+1), 'dropout' + str(idx+1)

            # 보통 affine -> batch norm -> activation -> dropout 순이다.
            self.layers[layer_name] = AffineLayer(self.params[w_name], self.params[b_name])
            
            if idx < num_layer - 1:
                self.layers[bn_name] = BatchNorm(self.params[g_name], self.params[d_name])
                self.layers[act_name] = ReLULayer()
                if self.use_dropout:
                    self.layers[dropout_name] = DropOut(self.dropout_ratio)
        
        self.last_layer = SoftmaxWithLoss()

    def weight_initialize(self):
        for idx in range(self.num_layer):
            w_name = 'W' + str(idx+1)
            if self.init_type == 'kaiming':
                self.params[w_name] *= np.sqrt(2 / self.layer_dim[idx])
            if self.init_type == 'xavier':
                self.params[w_name] *= np.sqrt(2 / (self.layer_dim[idx] + self.layer_dim[idx+1]))
            if isinstance(self.init_type, float):
                self.params[w_name] *= self.init_type

    def train_mode(self, train_flg):
        self.train_flg = train_flg

    def predict(self, x: np.ndarray):
        for layer in self.layers.values():
            if isinstance(layer, (DropOut, BatchNorm)):
                x = layer.forward(x, self.train_flg)
            else:
                x = layer.forward(x)
        return x
    
    def get_loss(self, x: np.ndarray, t: np.ndarray):
        x = self.predict(x)
        loss = self.last_layer.forward(x, t)

        weight_decay = 0
        for idx in range(self.num_layer):
            w_name = 'W' + str(idx+1)
            if self.decay_type == 'L1':
                weight_decay += self.decay_ratio * np.sum(np.abs(self.params[w_name]))
            if self.decay_type == 'L2':
                weight_decay += 0.5 * self.decay_ratio * np.sum(self.params[w_name] ** 2)
        
        loss += weight_decay
        return loss
    
    def get_accuracy(self, x: np.ndarray, t: np.ndarray):
        predictions = self.predict(x)
        predicted_labels = np.argmax(predictions, axis=1)
        ans_labels = np.argmax(t, axis=1)
        accuracy = np.sum(predicted_labels == ans_labels) / x.shape[0]
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
            g_name, d_name = 'gamma' + str(idx+1), 'delta' + str(idx+1)
            layer_name, bn_name = 'Affine' + str(idx+1), 'bn' + str(idx+1)

            grads[w_name] = self.layers[layer_name].dW
            if self.decay_type == 'L1':
                grads[w_name] += self.decay_ratio * sign(self.layers[layer_name].W)
            if self.decay_type == 'L2':
                grads[w_name] += self.decay_ratio * self.layers[layer_name].W

            grads[b_name] = self.layers[layer_name].db
            if idx < self.num_layer - 1:
                grads[g_name], grads[d_name] = self.layers[bn_name].dgamma, self.layers[bn_name].ddelta
        return grads
    
    def train(self, x: np.ndarray, t: np.ndarray, optimizer):
        grads = self.get_gradient(x, t)
        optimizer.update(self.params, grads)

if __name__ == '__main__':
    data_path = '/Users/ungsungko_master/Desktop/self/data'
    result_path = os.path.join(os.getcwd(), 'chapter_6/result')
    weight_path = os.path.join(result_path, 'weight/options')
    figure_path = os.path.join(result_path, 'figure/options')
    os.makedirs(weight_path, exist_ok=True)
    os.makedirs(figure_path, exist_ok=True)

    log = 'train MNIST dataset with relu for different training options (dropout, weight decay).\n\n'
    print('train MNIST dataset with relu for different training options (dropout, weight decay).\n')

    train_loss_dict, test_loss_dict = {}, {}
    train_accur_dict, test_accur_dict = {}, {}

    options = [None, 'L1', 'L2', 'dropout', ['L1', 'dropout'], ['L2, dropout']]

    hyper_param_set = {
        'epoch': 3000,
        'log_iter': 150,
        'lr': 0.001,
        'beta1': 0.9,
        'beta2': 0.999,
        'batch_size': 1000,
    }

    train_df = MNIST(root=data_path, train=True)
    test_df = MNIST(root=data_path, train=False)

    x_train, y_train = train_df.data.reshape(-1, 784) / 255.0, convert_to_one_hot(train_df.targets)
    x_test, y_test = test_df.data.reshape(-1, 784) / 255.0, convert_to_one_hot(test_df.targets)

    for idx, option in enumerate(options):
        log += f"### MNIST with option {option} ###"
        print(f"### MNIST with option {option} ###")

        if option == None:
            pkl_path = os.path.join(weight_path, 'mnist_with_no_option.pkl')
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as fid:
                    network = pickle.load(fid)
            else:
                network = MLP(decay_type=None, use_dropout=False)

        elif option in ['L1', 'L2']:
            pkl_path = os.path.join(weight_path, f'mnist_with_{option.lower()}_option.pkl')
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as fid:
                    network = pickle.load(fid)
            else:
                network = MLP(decay_type=option, use_dropout=False)

        elif option == 'dropout':
            pkl_path = os.path.join(weight_path, f'mnist_with_dropout_option.pkl')
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as fid:
                    network = pickle.load(fid)
            else:
                network = MLP(decay_type=None, use_dropout=True)

        elif isinstance(option, list):
            pkl_path = os.path.join(weight_path, f'mnist_with_{option[0].lower()}_dropout_option.pkl')
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as fid:
                    network = pickle.load(fid)
            else:
                network = MLP(decay_type=option[0], use_dropout=True)

        train_loss_dict[idx], test_loss_dict[idx] = [], []
        train_accur_dict[idx], test_accur_dict[idx] = [], []
        optimizer = Adam(lr=hyper_param_set['lr'], beta1=hyper_param_set['beta1'], beta2=hyper_param_set['beta2'])
        
        start_time = time.time()
        for epoch in tqdm(range(hyper_param_set['epoch'])):
            batch_mask = np.random.choice(x_train.shape[0], hyper_param_set['batch_size'], replace=False)
            x_batch, y_batch = x_train[batch_mask], y_train[batch_mask]

            network.train_mode(True)
            network.train(x_batch, y_batch, optimizer)

            if epoch % hyper_param_set['log_iter'] == 0:
                network.train_mode(False)

                loss = network.get_loss(x_batch, y_batch)
                accur = network.get_accuracy(x_batch, y_batch)
                train_loss_dict[idx].append(loss)
                train_accur_dict[idx].append(accur)

                log += f"\tAccuracy / Loss of train in epoch {epoch+1} / {hyper_param_set['epoch']} with option {option}: {accur:.3f} / {loss:.4f}\n"

                batch_mask = np.random.choice(x_test.shape[0], hyper_param_set['batch_size'], replace=False)
                x_batch, y_batch = x_test[batch_mask], y_test[batch_mask]

                loss = network.get_loss(x_batch, y_batch)
                accur = network.get_accuracy(x_batch, y_batch)
                test_loss_dict[idx].append(loss)
                test_accur_dict[idx].append(accur)

                log += f"\tAccuracy / Loss of test in epoch {epoch+1} / {hyper_param_set['epoch']} with option {option}: {accur:.3f} / {loss:.4f}\n"

        with open(pkl_path, 'wb') as fid:
            pickle.dump(network, fid)
        
        log += f"finished train with option {option}. (spent time: {time.time() - start_time:.4f})\n\n"
        print(f"finished train with option {option}. (spent time: {time.time() - start_time:.4f})\n")
    
    log += 'Train all finished.\n'

    labels = ['None', 'L1', 'L2', 'Dropout', 'L1+Dropout', 'L2+dropout']
    idx = np.arange(0, hyper_param_set['epoch'], hyper_param_set['log_iter'])

    train_accur_path = os.path.join(figure_path, 'train_accur_options.png')
    for i in range(len(options)):
        plt.plot(idx, train_accur_dict[i], label=labels[i])
    plt.xlabel('iter')
    plt.ylabel('accuracy')
    plt.legend()
    plt.title("Comparison of train accuracy with different options.")
    plt.savefig(train_accur_path)
    plt.close()

    test_accur_path = os.path.join(figure_path, 'test_accur_options.png')
    for i in range(len(options)):
        plt.plot(idx, test_accur_dict[i], label=labels[i])
    plt.xlabel('iter')
    plt.ylabel('accuracy')
    plt.legend()
    plt.title("Comparison of test accuracy with different options.")
    plt.savefig(test_accur_path)
    plt.close()

    train_loss_path = os.path.join(figure_path, 'train_loss_options.png')
    for i in range(len(options)):
        plt.plot(idx, train_loss_dict[i], label=labels[i])
    plt.xlabel('iter')
    plt.ylabel('loss')
    plt.legend()
    plt.title("Comparison of train loss with different options.")
    plt.savefig(train_loss_path)
    plt.close()

    test_loss_path = os.path.join(figure_path, 'test_loss_options.png')
    for i in range(len(options)):
        plt.plot(idx, test_loss_dict[i], label=labels[i])
    plt.xlabel('iter')
    plt.ylabel('loss')
    plt.legend()
    plt.title("Comparison of test loss with different options.")
    plt.savefig(test_loss_path)
    plt.close()