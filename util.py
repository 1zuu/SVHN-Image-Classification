import numpy as np 
from scipy.io import loadmat

from variables import*

def load_mat_file(mat_file):
    mat_data = loadmat(mat_file)
    x, y = mat_data['X'], mat_data['y']
    x = x.reshape(
                -1,
                input_shape[0],
                input_shape[1],
                input_shape[2]
                )
    y = y.reshape(-1,)
    return x, y

def load_data():
    X, Y = load_mat_file(train_dir)
    Xtest, Ytest = load_mat_file(test_dir)
    
    X = X/rescale
    Xtest = Xtest/rescale

    return X, Y, Xtest, Ytest