import os, sys, time, pickle
import numpy as np
import matplotlib.pyplot as plt

# sys.path.append(os.path.abspath(os.path.pardir)) 이렇게 하면 단순히 이 파일을 실행시킨 위치에서의 부모경로를 넣어버림
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from chapter_5.layers import *
from optimizer import *
from tqdm import tqdm
from collections import OrderedDict # 지금은 dict이 삽입순서를 보장하지만 예전에는 안그럼 (사실 python3.6 이상부터는 쓸 일이 없음)
from torchvision.datasets import MNIST

def convert_to_one_hot(x: np.ndarray, dim: int = 10):
    converted_x = np.zeros((x.shape[0], dim), np.float32)
    converted_x[np.arange(x.shape[0]), x] = 1.0
    return converted_x

# same with chapter_5 two_layer_net
class TwoLayerNet:
    def __init__(self, input_dim=784, hidden_dim=100, output_dim=10, weight_init_std=0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.params = {}
        self.params['W1'] = np.random.normal(loc=0.0, scale=weight_init_std, size=(input_dim, hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.normal(loc=0.0, scale=weight_init_std, size=(hidden_dim, output_dim))
        self.params['b2'] = np.zeros(output_dim)

        self.layers = OrderedDict()
        self.layers['Affine1'] = AffineLayer(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = ReLULayer()
        self.layers['Affine2'] = AffineLayer(self.params['W2'], self.params['b2'])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    def get_loss(self, x, t):
        predictions = self.predict(x)
        loss = self.last_layer.forward(predictions, t)
        return loss

    def get_accuracy(self, x, t):
        if x.ndim == 1:
            x = x.reshape(1, -1)
            t = t.reshape(1, -1)
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
            dout = layer.backward(dout=dout)
        
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        return grads
    
    def train(self, x, t, optimizer):
        grads = self.get_gradient(x, t)
        self.params = optimizer.update(self.params, grads)

if __name__ == '__main__':
    data_path = '/Users/ungsungko_master/Desktop/self/data'
    result_path = os.path.join(os.getcwd(), 'chapter_6/result')
    weight_path = os.path.join(result_path, 'weight/optim')
    figure_path = os.path.join(result_path, 'figure/optim')
    os.makedirs(weight_path, exist_ok=True)
    os.makedirs(figure_path, exist_ok=True)
    
    train_df = MNIST(root=data_path, train=True)
    test_df = MNIST(root=data_path, train=False)

    x_train, y_train = train_df.data.reshape(-1, 784), convert_to_one_hot(train_df.targets)
    x_test, y_test = test_df.data.reshape(-1, 784), convert_to_one_hot(test_df.targets)

    hyper_param_set = {
        'epoch': 2_001,
        'log_iter': 100,
        'lr': 0.001,
        'beta1': 0.9,
        'beta2': 0.999,
        'batch_size': 1000
    }

    sgd_optimizer = SGD(lr=hyper_param_set['lr'])
    momentum_optimizer = Momentum(lr=hyper_param_set['lr'], alpha=hyper_param_set['beta1'])
    adagrad_optimizer = AdaGrad(lr=hyper_param_set['lr'])
    rmsprop_optimizer = RMSProp(lr=hyper_param_set['lr'], p=hyper_param_set['beta2'])
    adam_optimizer = Adam(lr=hyper_param_set['lr'], beta1=hyper_param_set['beta1'], beta2=hyper_param_set['beta2'])

    optim_type = {
        'sgd': sgd_optimizer,
        'momentum': momentum_optimizer,
        'adagrad': adagrad_optimizer,
        'rmsprop': rmsprop_optimizer,
        'adam': adam_optimizer
    }

    log = ''
    train_loss_dict, train_accur_dict = OrderedDict(), OrderedDict()
    test_loss_dict, test_accur_dict = OrderedDict(), OrderedDict()
    for name, optimizer in optim_type.items():
        start_time = time.time()
        title = f"### start training mnist with optimizer {name} ###"
        print(title)
        log += f"{title}\n"

        train_loss_dict[name], train_accur_dict[name] = [], []
        test_loss_dict[name], test_accur_dict[name] = [], []

        pkl_path = os.path.join(weight_path, f"mnist_{name}.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as fid:
                network = pickle.load(fid)
        else:
            network = TwoLayerNet()
        
        for epoch in tqdm(range(hyper_param_set['epoch'])):
            batch_mask = np.random.choice(x_train.shape[0], hyper_param_set['batch_size'], replace=False)
            x_batch, y_batch = x_train[batch_mask], y_train[batch_mask]

            network.train(x_batch, y_batch, optimizer=optimizer)

            if epoch % hyper_param_set['log_iter'] == 0:
                accuracy = network.get_accuracy(x_batch, y_batch)
                loss = network.get_loss(x_batch, y_batch)

                train_accur_dict[name].append(accuracy)
                train_loss_dict[name].append(loss)
            
                log += f"\tAccuracy / Loss of train in epoch {epoch+1} / {hyper_param_set['epoch']} with optimizer {name} : {accuracy:.2f} / {loss:.4f}\n"
                
                batch_mask = np.random.choice(x_test.shape[0], hyper_param_set['batch_size'], replace=False)
                x_batch, y_batch = x_test[batch_mask], y_test[batch_mask]
                accuracy = network.get_accuracy(x_batch, y_batch)
                loss = network.get_loss(x_batch, y_batch)

                test_accur_dict[name].append(accuracy)
                test_loss_dict[name].append(loss)

                log += f"\tAccuracy / Loss of test in epoch {epoch+1} / {hyper_param_set['epoch']} with optimizer {name} : {accuracy:.2f} / {loss:.4f}\n"
                
        with open(pkl_path, 'wb') as fid:
            pickle.dump(network, fid)
        log += f"finished train with optimizer {name}. (spent time: {time.time() - start_time:.4f})\n\n"

    log += 'Train all finished.\n'
    
    idx = range(0, hyper_param_set['epoch'], hyper_param_set['log_iter'])
    for name in optim_type.keys():
        train_path = os.path.join(figure_path, f"{name}_mnist_train.png")
        test_path = os.path.join(figure_path, f"{name}_mnist_test.png")

        fig, ax = plt.subplots()
        ax.plot(idx, train_accur_dict[name], color='r', label='accuracy')
        ax.set_xlabel('iter')
        ax.set_ylabel('accuracy')

        ax2 = ax.twinx()
        ax2.plot(idx, train_loss_dict[name], color='gray', label='loss')
        ax2.set_ylabel('loss')

        ax.legend(ax.get_lines() + ax2.get_lines(), [obj.get_label() for obj in ax.get_lines() + ax2.get_lines()], loc='best')

        plt.title(f"Train result with optimizer {name}")
        plt.savefig(train_path)
        plt.close()

        fig, ax = plt.subplots()
        ax.plot(idx, test_accur_dict[name], color='r', label='accuracy')
        ax.set_xlabel('iter')
        ax.set_ylabel('accuracy')

        ax2 = ax.twinx()
        ax2.plot(idx, test_loss_dict[name], color='gray', label='loss')
        ax2.set_ylabel('loss')

        ax.legend(ax.get_lines() + ax2.get_lines(), [obj.get_label() for obj in ax.get_lines() + ax2.get_lines()], loc='best')

        plt.title(f"Test result with optimizer {name}")
        plt.savefig(test_path)
        plt.close()
    
    train_path = os.path.join(figure_path, 'train_loss_all.png')
    test_path = os.path.join(figure_path, 'test_loss_all.png')

    for name in optim_type.keys():
        plt.plot(idx, train_loss_dict[name], label=name)
    plt.xlabel('iter')
    plt.ylabel('loss')
    plt.title('Comparison train result with all optimzers')
    plt.legend()
    plt.savefig(train_path)
    plt.close()
        
    for name in optim_type.keys():
        plt.plot(idx, test_loss_dict[name], label=name)
    plt.xlabel('iter')
    plt.ylabel('loss')
    plt.title('Comparison test result with all optimzers')
    plt.legend()
    plt.savefig(test_path)
    plt.close()

    log += 'All finished.'

    log_path = os.path.join(result_path, 'log._optim.txt')
    with open(log_path, 'w') as fid:
        fid.write(log)