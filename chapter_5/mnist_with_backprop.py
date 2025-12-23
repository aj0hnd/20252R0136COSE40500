import os, time
import pickle
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets
from two_layer_net import TwoLayerNet

def convert_to_onehot(arr: np.ndarray, target_dim: int = 10):
    converted = np.zeros((arr.shape[0], target_dim))
    converted[np.arange(arr.shape[0]), arr] = 1.
    return converted

    
if __name__ == '__main__':
    data_path = '/Users/ungsungko_master/Desktop/self/data'
    log_path = os.path.join(os.getcwd(), 'chapter_5/result/log.txt')
    weight_path = os.path.join(os.getcwd(), 'chapter_5/result/weight.pkl')
    train_path = os.path.join(os.getcwd(), 'chapter_5/result/train.png')
    test_path = os.path.join(os.getcwd(), 'chapter_5/result/test.png')
    visualize_path = os.path.join(os.getcwd(), 'chapter_5/result/visualize.png')

    train_df = datasets.MNIST(root=data_path, train=True)
    x_train, y_train = train_df.data.numpy().reshape(-1, 784) / 255.0, convert_to_onehot(train_df.targets.numpy())

    test_df = datasets.MNIST(root=data_path, train=False)
    x_test, y_test = test_df.data.numpy().reshape(-1, 784) / 255.0, convert_to_onehot(test_df.targets.numpy())

    lr = 0.01
    batch_size, train_size, test_size, step_num = 500, x_train.shape[0], x_test.shape[0], 10_000

    if os.path.exists(weight_path): 
        print("Found pre-trained network. Start with pre-trained weights.")
        with open(weight_path, 'rb') as fid:
            network = pickle.load(fid)
    else:
        print("Cannot find pre-trained network. Train from scratch.")
        network = TwoLayerNet(input_dim=784, hidden_dim=50, output_dim=10)

    fid = open(log_path, 'w')
    train_loss_list, train_accur_list, test_loss_list, test_accur_list = [], [], [], []
    start_time = time.time()

    for iter_num in range(step_num + 1):
        batch_mask = np.random.choice(train_size, batch_size, replace=False)
        x_batch, y_batch = x_train[batch_mask], y_train[batch_mask]

        network.train(x_batch, y_batch, lr=lr)

        if iter_num % 500 == 0:
            loss, accur = network.get_loss(x_batch, y_batch), network.get_accuracy(x_batch, y_batch)

            train_log = f"train loss / accuracy at {iter_num} / {step_num} : {loss} / {accur}"
            print(train_log)
            train_loss_list.append(loss)
            train_accur_list.append(accur)

            batch_mask = np.random.choice(test_size, 100, replace=False)
            x_batch, y_batch = x_test[batch_mask], y_test[batch_mask]

            loss, accur = network.get_loss(x_batch, y_batch), network.get_accuracy(x_batch, y_batch)
            test_log = f"test loss / accuracy at {iter_num} / {step_num} : {loss} / {accur}"
            print(test_log + '\n')
            test_loss_list.append(loss)
            test_accur_list.append(accur)

            fid.writelines([train_log + '\n', test_log + '\n'])
    
    fid.close()

    with open(weight_path, 'wb') as fid:
        pickle.dump(network, fid)
    
    x = np.arange(0, step_num + 1, 500)
    
    # save train results
    fig, ax = plt.subplots()
    ax.plot(x, train_accur_list, color='r', label='accuracy')
    ax.set_xlabel('iter')
    ax.set_ylabel('accuracy')

    ax2 = ax.twinx()
    ax2.plot(x, train_loss_list, color='gray', linestyle='--', label='loss')
    ax2.set_ylabel('loss')

    ax.legend(ax.get_lines() + ax2.get_lines(), [obj.get_label() for obj in ax.get_lines() + ax2.get_lines()], loc='best')
    plt.title('Train results')
    plt.savefig(train_path)    

    # save test results
    fig, ax = plt.subplots()
    ax.plot(x, test_accur_list, color='r', label='accuracy')
    ax.set_xlabel('iter')
    ax.set_ylabel('accuracy')

    ax2 = ax.twinx()
    ax2.plot(x, test_loss_list, color='gray', linestyle='--', label='loss')
    ax2.set_ylabel('loss')

    ax.legend(ax.get_lines() + ax2.get_lines(), [obj.get_label() for obj in ax.get_lines() + ax2.get_lines()], loc='best')
    plt.title('Test results')
    plt.savefig(test_path)

    # visualize
    batch_mask = np.random.choice(test_size, 10, replace=False)
    test_batch, test_labels = x_test[batch_mask], np.argmax(y_test[batch_mask], axis=1)
    predictions = network.predict(test_batch)
    predictioned_labels = np.argmax(predictions, axis=1)

    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(20, 6))
    for i in range(2):
        for j in range(5):
            idx = 5 * i + j
            color = 'blue' if predictioned_labels[idx] == test_labels[idx] else 'red'
            ax[i][j].imshow(test_batch[idx].reshape(28, 28), cmap='gray')
            ax[i][j].set_title(f"predicted label : {int(predictioned_labels[idx])}\nanswer : {int(test_labels[idx])}", color=color)
            ax[i][j].axis('off')
    fig.suptitle('Prediction Visualization', fontsize=20)
    plt.tight_layout()
    plt.savefig(visualize_path)

    spend_time = time.time() - start_time
    print(f"All processes finished. Time spent {spend_time:.4f} for {step_num} epochs with batch {batch_size}.")
