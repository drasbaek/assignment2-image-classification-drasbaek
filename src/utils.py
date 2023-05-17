""" utils.py
Author: 
    Anton Drasbæk Schiønning (202008161), GitHub: @drasbaek

Desc:
    Contains support functions for the classify.py script.
"""

# load packages
import numpy as np
import argparse
from model_space import *
from pathlib import Path
import os
import joblib

def define_paths():
    '''
    Define paths for input and output data.

    Returns:
    -   inpath (pathlib.PosixPath): Path to input data.
    -   model_outpath (pathlib.PosixPath): Path to output folder for models.
    -   report_outpath (pathlib.PosixPath): Path to output folder for reports.
    '''

    # define paths
    path = Path(__file__)

    # define input dir
    inpath = path.parents[1] / "data"

    # define output dir for models
    model_outpath = path.parents[1] / "models"

    # define output dir for reports
    report_outpath = path.parents[1] / "out"

    return inpath, model_outpath, report_outpath


def arg_parse():
    '''
    Parse command line arguments.
    It is possible to specify which classifiers to train.

    Returns:
    -   args (argparse.Namespace): Parsed arguments.
    '''
    
    # define parser
    parser = argparse.ArgumentParser(description='Train a classifier on the CIFAR10 dataset')

    # add argument
    parser.add_argument('-c', '--classifiers', nargs='+', default=['logistic', 'neural'], help='Classifiers to train')

    # parse arguments
    args = parser.parse_args()

    return args


def input_checker(args, model_space):
    '''
    Check if the input arguments are correct and return error message if not.
    
    Args:
    -   args (argparse.Namespace): Parsed arguments.
    -   model_space (dict): Dictionary with model space.
    '''

    # check if classifiers argument is correct
    if not set(args.classifiers).issubset(set(model_space.keys())):

        # create error message
        error_message = f"At least one classifier argument is incorrect. Please choose one or more from the following classifiers: {list(models.keys())}"

        # raise error
        raise ValueError(error_message)


def load_data(inpath):
    '''
    Loads preprocessed Cifar10 data from data folder. Also contains labels for data.

    Args:
    -   inpath (pathlib.PosixPath): Path to data folder.

    Returns:
    -   X_train (np.array): Training data.
    -   y_train (np.array): Training labels.
    -   X_test (np.array): Test data.
    -   y_test (np.array): Test labels.
    -   labels (list): List with labels for data.
    '''

    # load data from data folder
    data = np.load(os.path.join(inpath, "cifar10_preprocessed.npz"))

    # split data into train and test
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']

    # define labels for data
    labels = ["airplane", 
          "automobile", 
          "bird", 
          "cat", 
          "deer", 
          "dog", 
          "frog", 
          "horse", 
          "ship", 
          "truck"]

    return X_train, y_train, X_test, y_test, labels


def save_report(report, report_outpath, model, params):
    '''
    Save classification report to output folder with name by model type.

    Args:
    -   report (str): Classification report.
    -   report_outpath (pathlib.PosixPath): Path to output folder.
    -   model (str): Model type.
    -   params (dict): Dictionary with model parameters.
    '''

    # create string with report name
    report_name = f'{model}_report.txt'

    # write report to file with specification for model at the top to the output folder
    with open(os.path.join(report_outpath, report_name), 'w') as f:
        f.write(f'{model} with parameters {params} \n' + report)


def save_model(best_clf, model, model_outpath):
    '''
    Save model to models folder with name by model type.

    Args:
    -   best_clf (sklearn.model): Best model.
    -   model (str): Model type.
    -   model_outpath (pathlib.PosixPath): Path to models folder.
    '''

    # define model name
    model_name = f"{model}_model"

    # dump model
    joblib.dump(best_clf, os.path.join(model_outpath, f"{model_name}.joblib"))