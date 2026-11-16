import os
import pickle
import numpy as np
from glob import glob

""" This script implements the functions for reading data.
"""

def load_data(data_dir):
    """ Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches are stored.
    
    Returns:
        x_train: An numpy array of shape [50000, 3072]. 
        (dtype=np.float32)
        y_train: An numpy array of shape [50000,]. 
        (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072]. 
        (dtype=np.float32)
        y_test: An numpy array of shape [10000,]. 
        (dtype=np.int32)
    """
    # get training paths
    train_paths = glob(f'{data_dir}/data_batch*')

    # get test path
    test_path = f'{data_dir}/test_batch'

    # load in as pickle
    x_train = []
    y_train = []
    for path in train_paths:
        with open(path, 'rb') as fo:
            data = pickle.load(fo, encoding='bytes')
            x_train.append(data[b'data'])
            y_train.extend(data[b'labels'])

    # concat data
    x_train = np.vstack(x_train)
    y_train = np.array(y_train, dtype=int)

    # do same for test
    with open(test_path, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
        x_test = data[b'data']
        y_test = np.array(data[b'labels'], dtype=int)



    return x_train, y_train, x_test, y_test

def train_vaild_split(x_train, y_train, split_index=45000):
    """ Split the original training data into a new training dataset
        and a validation dataset.
    
    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        split_index: An integer.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """
    x_train_new = x_train[:split_index]
    y_train_new = y_train[:split_index]
    x_valid = x_train[split_index:]
    y_valid = y_train[split_index:]

    return x_train_new, y_train_new, x_valid, y_valid