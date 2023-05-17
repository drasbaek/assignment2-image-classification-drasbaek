""" preprocess.py
Author: 
    Anton Drasbæk Schiønning (202008161), GitHub: @drasbaek

Desc:
    This script loads and preprocesses the CIFAR10 dataset from keras datasets.
    The data is then saved as a compressed numpy array in the data folder to be used for the classify.py script.

Usage:
    $ python src/preprocess.py
"""

# load packages
import os
from tensorflow.keras.datasets import cifar10
import cv2
import numpy as np


def load_cifar10():
    '''
    Load cifar10 dataset from keras datasets

    Returns:
        X_train: training data
        y_train: training labels
        X_test: test data
        y_test: test labels
    '''
    # load data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    return X_train, y_train, X_test, y_test


def preprocess_data(X_train, X_test):
    '''
    Preprocesses data by converting to greyscale, scaling and flattening

    Args:
        X_train: training data
        X_test: test data
    
    Returns:
        X_train_dataset: preprocessed training data
        X_test_dataset: preprocessed test data
    '''

    # convert to greyscale
    X_train_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train])
    X_test_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test])

    # scale data
    X_train_scaled = X_train_grey / 255.0
    X_test_scaled = X_test_grey / 255.0

    # reshape/flatten data
    nsamples, nx, ny = X_train_scaled.shape
    X_train_dataset = X_train_scaled.reshape((nsamples,nx*ny))  

    nsamples, nx, ny = X_test_scaled.shape
    X_test_dataset = X_test_scaled.reshape((nsamples,nx*ny))

    return X_train_dataset, X_test_dataset


def save_data(X_train_dataset, y_train, X_test_dataset, y_test):
    '''
    Exports data as compressed numpy array

    Args:
        X_train_dataset: preprocessed training data
        y_train: training labels
        X_test_dataset: preprocessed test data
        y_test: test labels
    '''
    # save data as npz
    np.savez_compressed(os.path.join(os.getcwd(), 'data/cifar10_preprocessed.npz'), X_train=X_train_dataset, y_train=y_train, X_test=X_test_dataset, y_test=y_test)


def main():
    # load cifar10 dataset
    X_train, y_train, X_test, y_test = load_cifar10()
    
    # preprocess dataset
    X_train_dataset, X_test_dataset = preprocess_data(X_train, X_test)

    # save data as npz
    save_data(X_train_dataset, y_train, X_test_dataset, y_test)

# run main
if __name__ == '__main__':
    main()
