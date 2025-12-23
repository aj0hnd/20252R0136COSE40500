import os, sys, time, pickle
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import OrderedDict
from torchvision.datasets import MNIST

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))
from optimizer import SGD
from options import BatchNorm, DropOut
from chapter_5.layers import AffineLayer, ReLULayer, SoftmaxWithLoss

def convert_to_one_hot(x: np.ndarray):
    out = np.zeros((x.shape[0], 10))
    out[np.arange(x.shape[0]), x] = 1.0
    return out


class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, num_layer, 
                 use_norm=True, use_decay=True, use_dropout=True, decay_ratio=0.001, dropout_ratio=0.5):
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.num_layer = num_layer
        self.layer_dim = [input_dim] + [hidden_dim] * (num_layer - 1) + [output_dim]

        self.use_norm = use_norm
        self.use_decay = use_decay
        self.use_dropout = use_dropout

        self.decay_ratio = decay_ratio
        self.dropout_ratio = dropout_ratio

        self.train_flg = None

        self.params = OrderedDict()
        for idx in range(self.num_layer):
            w_name, b_name = 'W' + str(idx+1), 'b' + str(idx+1)
            self.params[w_name] = np.random.normal(0, 1.0, size=(self.layer_dim[idx], self.layer_dim[idx+1]))
            self.params[b_name] = np.zeros(self.layer_dim[idx+1])

            if self.use_norm and idx < self.num_layer - 1:
                g_name, d_name = 'gamma' + str(idx+1), 'delta' + str(idx+1)
                self.params[g_name] = np.ones((1, self.layer_dim[idx+1]))
                self.params[d_name] = np.zeros((1, self.layer_dim[idx+1]))

        self.layers = OrderedDict()
        for idx in range(self.num_layer):
            w_name, b_name = 'W' + str(idx+1), 'b' + str(idx+1)
            layer_name = 'Affine' + str(idx+1)
            self.layers[layer_name] = AffineLayer(self.params[w_name], self.params[b_name])

            if self.use_norm and idx < self.num_layer - 1:
                g_name, d_name = 'gamma' + str(idx+1), 'delta' + str(idx+1)
                bn_name = 'BatchNorm' + str(idx+1)
                self.layers[bn_name] = BatchNorm(self.params[g_name], self.params[d_name])
            
            if idx < self.num_layer - 1:
                relu_name = 'Relu' + str(idx+1)
                self.layers[relu_name] = ReLULayer()
            
            if self.use_dropout and idx < self.num_layer - 1:
                dropout_name = 'Dropout' + str(idx+1)
                self.layers[dropout_name] = DropOut(self.dropout_ratio)
        
        self.last_layer = SoftmaxWithLoss()
        
        self.weight_initialize()
    
    def weight_initialize(self):
        for layer_name, layer in self.layers.items():
            if isinstance(layer, AffineLayer):
                layer_idx = int(layer_name.split("Affine")[1]) - 1
                layer.W *= np.sqrt(2.0 / self.layer_dim[layer_idx])

    def trainable(self, train_flg):
        self.train_flg = train_flg
    
    def predict(self, x: np.ndarray):
        for layer in self.layers.values():
            if isinstance(layer, (BatchNorm, DropOut)):
                x = layer.forward(x, self.train_flg)
            else:
                x = layer.forward(x)
        return x
    
    def get_loss(self, x: np.ndarray, t: np.ndarray):
        x = self.predict(x)
        loss = self.last_layer.forward(x, t)
        return loss
    
    def get_accuracy(self, x: np.ndarray, t: np.ndarray):
        predicted = self.predict(x)
        
        predicted_labels = np.argmax(predicted, axis=1)
        ans_labels = np.argmax(t, axis=1)
        
        accuracy = np.sum(predicted_labels == ans_labels) / x.shape[0]
        return accuracy
    
    def get_gradient(self, x: np.ndarray, t: np.ndarray):
        self.get_loss(x, t)

        dout = 1
        dout = self.last_layer.backward(dout)

        for layer in reversed(list(self.layers.values())):
            dout = layer.backward(dout)
        
        grads = OrderedDict()
        for idx in range(self.num_layer):
            w_name, b_name = 'W' + str(idx+1), 'b' + str(idx+1)
            layer_name = 'Affine' + str(idx+1)

            grads[w_name], grads[b_name] = self.layers[layer_name].dW, self.layers[layer_name].db

            if self.use_norm and idx < self.num_layer - 1:
                g_name, d_name = 'gamma' + str(idx+1), 'delta' + str(idx+1)
                bn_name = 'BatchNorm' + str(idx+1)

                grads[g_name], grads[d_name] = self.layers[bn_name].dgamma, self.layers[bn_name].ddelta
        return grads
    
    def train(self, x: np.ndarray, t: np.ndarray, optimizer):
        grads = self.get_gradient(x, t)
        optimizer.update(self.params, grads)

if __name__ == '__main__':
    data_path = '/Users/ungsungko_master/Desktop/self/data'
    result_path = os.path.join(os.getcwd(), 'chapter_6/result')
    weight_path = os.path.join(result_path, 'weight/hyperparam')
    figure_path = os.path.join(result_path, 'figure/hyperparam')
    
    os.makedirs(weight_path, exist_ok=True)
    os.makedirs(figure_path, exist_ok=True)

    train_df = MNIST(root=data_path, train=True)
    test_df = MNIST(root=data_path, train=False)

    x_train, y_train = train_df.data.reshape(-1, 784) / 255.0, convert_to_one_hot(train_df.targets)
    x_test, y_test = test_df.data.reshape(-1, 784) / 255.0, convert_to_one_hot(test_df.targets)

    validation_ratio = 0.2
    validation_num = int(x_train.shape[0] * validation_ratio)

    x_valid, y_valid = x_train[validation_num:], y_train[validation_num:]
    x_train, y_train = x_train[:validation_num], y_train[:validation_num]

    hyperparam_set = {
        'train_batch': 1_000, 
        'test_batch': 2_000,
        'num_layer': 5,
        'lr': 0.001, # need to optimize!
        'beta1': 0.9,
        'beta2': 0.999,
        'decay_ratio': 0.001, # need to optimize!
        'dropout_ratio': 0.5,
        'epoch': 500,
        'log_iter': 25,
    }

    lr_candidate = [10 ** (i) for i in range(-6, 2)]
    decay_candidate = [10 ** (i) for i in range(-3, 3)]

    logs = ''
    fig, ax = plt.subplots(nrows=len(lr_candidate), ncols=len(decay_candidate), figsize=(20, 20))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=1.0)

    for i, lr in enumerate(lr_candidate):
        for j, decay_ratio in enumerate(decay_candidate):
            logs += f"Start training with # {i * len(decay_candidate) + j + 1} / {len(lr_candidate) * len(decay_candidate)} lr = {lr} / decay = {decay_ratio}\n\n"
            print(f"Start training with # {i * len(decay_candidate) + j + 1} / {len(lr_candidate) * len(decay_candidate)} lr = {lr} / decay = {decay_ratio}")

            start_time = time.time()

            hyperparam_set['lr'] = lr
            hyperparam_set['decay_ratio'] = decay_ratio

            train_accur_list, valid_accur_list = [], []

            pkl_path = os.path.join(weight_path, f"lr_{lr}_decay_{decay_ratio}.pkl")
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as fid:
                    network = pickle.load(fid)
            else:
                network = MLP(784, 50, 10, hyperparam_set['num_layer'], decay_ratio=hyperparam_set['decay_ratio'])

            optimizer = SGD(lr=hyperparam_set['lr'])
            
            for epoch in tqdm(range(hyperparam_set['epoch'])):
                batch_mask = np.random.choice(x_train.shape[0], hyperparam_set['train_batch'], replace=False)
                x_batch, y_batch = x_train[batch_mask], y_train[batch_mask]

                network.trainable(True)
                network.train(x_batch, y_batch, optimizer)

                if epoch % hyperparam_set['log_iter'] == 0:
                    network.trainable(False)
                    
                    accuracy = network.get_accuracy(x_batch, y_batch)
                    train_accur_list.append(accuracy)
                    logs += f"Train accuracy with lr {lr} / decay {decay_ratio} : {accuracy:.3f}\n"

                    accuracy = network.get_accuracy(x_valid, y_valid)
                    valid_accur_list.append(accuracy)

                    logs += f"Validation accuracy with lr {lr} / decay {decay_ratio} : {accuracy:.3f}\n"
            
            idx = np.arange(0, hyperparam_set['epoch'], hyperparam_set['log_iter'])
            ax[i][j].plot(idx, train_accur_list, color='red', label='train')
            ax[i][j].plot(idx, valid_accur_list, color='blue', label='valid')
            ax[i][j].set_xlabel('iter')
            ax[i][j].set_ylabel('accuracy')
            ax[i][j].set_title(f"test-{i*len(lr_candidate)+j+1}\n({lr} / {decay_ratio})")
            ax[i][j].legend(loc='upper left')

            with open(pkl_path, 'wb') as fid:
                pickle.dump(network, fid)

            spend_min, spend_sec = int((time.time() - start_time) // 60), (time.time() - start_time) % 60
            logs += f"Finished with lr {lr} / decay {decay_ratio} with {spend_min}M {spend_sec}s.\n\n"
            print(f"Finished with lr {lr} / decay {decay_ratio} with {spend_min}M {spend_sec:.2f}s.\n")
    
    plt.suptitle('hyperparameter optimization with MNIST')

    final_path = os.path.join(figure_path, 'optimized.png')
    plt.savefig(final_path)
    plt.close()

    log_path = os.path.join(result_path, 'log_hyperparam.txt')
    with open(log_path, 'w') as fid:
        fid.write(logs)
    