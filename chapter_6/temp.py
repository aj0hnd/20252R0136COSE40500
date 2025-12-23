import os
import pickle
import numpy as np
from compare_batchnorm_with_mnist import MLP

result_path = os.path.join(os.getcwd(), 'chapter_6/result')
weight_path = os.path.join(result_path, 'weight/batchnorm')

pkl_path = os.path.join(weight_path, 'mnist_with_bn_True_5.pkl')
with open(pkl_path, 'rb') as fid:
    network = pickle.load(fid)

print(network.params['g1'])