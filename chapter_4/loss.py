import numpy as np
from torchvision import datasets

def get_mse(predictions: np.ndarray, labels: np.ndarray):
    assert predictions.shape == labels.shape
    if len(predictions.shape) == 1:
        predictions = predictions.reshape(1, -1)
        labels = labels.reshape(1, -1)

    mse = 0.5 * np.sum((predictions - labels) ** 2) / predictions.shape[0]
    return mse

def get_ce(predictions: np.ndarray, labels: np.ndarray, eps: float = 1e-5):
    assert predictions.shape == labels.shape
    if len(predictions.shape) == 1:
        predictions = predictions.reshape(1, -1)
        labels = labels.reshape(1, -1)

    ce = -np.sum(labels * np.log(predictions + eps)) / predictions.shape[0]
    return ce

if __name__ == '__main__':
    data_path = '/Users/ungsungko_master/Desktop/self/data'

    train_df = datasets.MNIST(
        root=data_path,
        train=True,
        download=False
    )
    test_df = datasets.MNIST(
        root=data_path,
        train=False,
        download=False
    )

    x_train, y_train = train_df.data.numpy(), train_df.targets.numpy()
    
    train_size, batch_size = x_train.shape[0], 10
    batch_mask = np.random.choice(train_size, batch_size, replace=False)
    print(f"index of selected batch: {batch_mask}")

    x_batch, y_batch = x_train[batch_mask], y_train[batch_mask]
    print(f"shape of x, y batch: {x_batch.shape}, {y_batch.shape}\n")

    random_values = np.random.rand(batch_size)
    print(f"MSE error from random prediction vs. label: {get_mse(predictions=random_values, labels=y_batch):.4f}")
    print(f"CE error from random prediction vs. label: {get_ce(predictions=random_values, labels=y_batch):.4f}")