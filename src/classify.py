""" classify.py
Author: 
    Anton Drasbæk Schiønning (202008161), GitHub: @drasbaek

Desc:
    This script trains and evaluates a single or multiple sklearn-based classifiers on the CIFAR10 dataset.
    It is setup to work with GridSearchCV for parameter tuning.
    Note: Utilizes the model_space.py script for defining the model space and support functions from the utils.py script.

Usage:
    $ python src/classify.py
"""

# load packages
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import logging
import os
import sys
sys.path.append("..")
from model_space import *
from utils import *

class Classifier:
    '''
    Class for training and evaluating a classifier.
    It is setup to work with GridSearchCV and can be used for any classifier in the sklearn library.

    Args:
    -   X_train(): training data
    -   y_train: training labels
    -   X_test: test data
    -   y_test: test labels
    -   labels: list of labels
    -   classifier_type: type of classifier
    -   param_grid: parameter grid for classifier
    
    Methods:
    -   create: create classifier
    -   train: train classifier
    -   evaluate: evaluate classifier

    Returns:
    -   classifier: trained classifier
    -   report: classification report
    -   params: best parameters
    '''

    # initialize classifier parameters
    def __init__(self, X_train, y_train, X_test, y_test, labels, classifier_type, param_grid):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.labels = labels
        self.classifier_type = classifier_type
        self.param_grid = param_grid
        self.classifier = None

    # create classifier
    def create(self):
        # log creation
        logging.info("(1/3) Creating classifier")

        # define the classifier
        self.classifier = self.classifier_type
        
        # define the grid search using param_grid and standard parameters
        self.classifier = GridSearchCV(self.classifier, self.param_grid, cv=3, scoring = "accuracy", verbose=3)

    # train classifier
    def train(self):
        # log training
        logging.info("(2/3) Training classifier")

        # train classifier
        self.classifier.fit(self.X_train, self.y_train.ravel())

    # evaluate classifier
    def evaluate(self):
        # log evaluation
        logging.info("(3/3) Evaluating classifier")

        # evaluate classifier
        y_pred = self.classifier.predict(self.X_test)

        # create classification report
        report = classification_report(self.y_test, y_pred, target_names=self.labels)

        # extract parameters
        params = self.classifier.best_params_

        return self.classifier, report, params


def main():
    # define paths
    inpath, model_outpath, report_outpath = define_paths()

    # config logging
    logging.basicConfig(format='%(message)s', level=logging.INFO)

    # parse arguments
    args = arg_parse()

    # check if input arguments are correct
    input_checker(args, model_space)

    # load data
    X_train, y_train, X_test, y_test, labels = load_data(inpath)

    for model in args.classifiers:
        # log start of classifier
        logging.info(" " + "\n" + f" ====== Starting classifier: {model} ({args.classifiers.index(model) + 1} out of {len(args.classifiers)}) ======")

        # identify classifier and param_grid
        classifier_type = model_space[model]["type"]
        param_grid = model_space[model]["param_grid"]
        
        # define parameters
        clf = Classifier(X_train, y_train, X_test, y_test, labels, classifier_type = classifier_type, param_grid = param_grid)

        # create classifier
        clf.create()

        # train classifier
        clf.train()

        # evaluate classifier
        best_clf, report, params = clf.evaluate()

        # save classification report
        save_report(report, report_outpath, model, params)

        # save model
        save_model(best_clf, model, model_outpath)

    # log end of program and where to find results
    logging.info("\n" + " " + f"====== Finished all classifiers ======" + "\n" + " ")

    # log what folder to find results in
    logging.info(f"Reports can be found at {report_outpath}")
    logging.info(f"Models can be found at {model_outpath}")

# run main
if __name__ == '__main__':
    main()

