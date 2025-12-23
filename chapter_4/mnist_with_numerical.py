import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
from two_layer_net import TwoLayerNet

def transform_to_one_hot(labels: np.ndarray, output_dim: int = 10):
    transformed_labels = np.zeros((labels.shape[0], output_dim))
    transformed_labels[np.arange(labels.shape[0]), labels] = 1.
    return transformed_labels

if __name__ == '__main__':
    data_path = '/Users/ungsungko_master/Desktop/self/data'
    log_path = os.path.join(os.getcwd(), 'chapter_4/result/log.txt')
    weight_path = os.path.join(os.getcwd(), 'chapter_4/result/weight.npz')
    train_path = os.path.join(os.getcwd(), 'chapter_4/result/train.png')
    test_path = os.path.join(os.getcwd(), 'chapter_4/result/test.png')

    train_df = datasets.MNIST(root=data_path, train=True)
    test_df = datasets.MNIST(root=data_path, train=False)
    
    x_train, y_train = train_df.data.numpy().reshape(-1, 784), transform_to_one_hot(train_df.targets.numpy())
    x_test, y_test = test_df.data.numpy().reshape(-1, 784), transform_to_one_hot(test_df.targets.numpy())
    
    network = TwoLayerNet(input_dim=784, hidden_dim=50, output_dim=10)
    if os.path.exists(weight_path):
        previous_weight = np.load(weight_path)
        for param in previous_weight.files:
            network.params[param] = previous_weight[param]

    train_loss_list, test_loss_list = [], []
    train_accur_list, test_accur_list = [], []
    lr, iter_num = 0.1, 1_000
    train_size, batch_size = x_train.shape[0], 100

    fid = open(log_path, 'w')

    for i in range(iter_num + 1):
        batch_mask = np.random.choice(train_size, batch_size, replace=False)
        x_batch, y_batch = x_train[batch_mask], y_train[batch_mask]

        network.train(inputs=x_batch, labels=y_batch, lr=lr)

        if i % 10 == 0:
            train_loss = network.get_loss(x_batch, y_batch)
            train_loss_list.append(train_loss)
            train_accur = network.get_accuracy(x_batch, y_batch)
            train_accur_list.append(train_accur)

            test_loss = network.get_loss(x_test, y_test)
            test_loss_list.append(test_loss)
            test_accur = network.get_accuracy(x_test, y_test)
            test_accur_list.append(test_accur)
            
            train_info = f"train loss / accuracy at epoch {i+1}/{iter_num} : {train_loss} / {train_accur}\n"
            test_info = f"test loss / accuracy at epoch {i+1}/{iter_num} : {test_loss} / {test_accur}\n"
            fid.writelines([train_info, test_info])
            fid.write('\n')
            print(train_info, end="")
            print(test_info, end="")

    fid.close()
    
    x = np.arange(0, iter_num + 1, 100)
    # save train
    fig, ax = plt.subplots()

    ax.plot(x, train_accur_list, color='r', label='accuracy')
    ax.set_xlabel('iter')
    ax.set_ylabel('accuracy')
    ax.set_ybound(lower=0.0, upper=1.0)    

    ax2 = ax.twinx()
    ax2.plot(x, train_loss_list, color='gray', linestyle='--')
    ax2.set_ylabel('loss')

    plt.title('Train result.')
    plt.legend()
    plt.savefig(train_path)

    # save test
    fig, ax = plt.subplots()

    ax.plot(x, test_accur_list, color='r', label='accuracy')
    ax.set_xlabel('iter')
    ax.set_ylabel('accuracy')
    ax.set_ybound(lower=0.0, upper=1.0)    

    ax2 = ax.twinx()
    ax2.plot(x, test_loss_list, color='gray', linestyle='--')
    ax2.set_ylabel('loss')

    plt.title('Test result.')
    plt.legend()
    plt.savefig(test_path)

    np.savez(weight_path, **network.params)