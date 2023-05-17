# Assignment 2: Image Classification

## Repository Overview
1. [Description](#description)
2. [Repository Tree](#tree)
3. [Usage](#usage)
4. [Results](#results)
5. [Discussion](#discussion)


## Description <a name="description"></a>
This repository includes the solution by *Anton Drasbæk Schiønning (202008161)* to assignment 2 in the course "Visual Analytics" at Aarhus University.

It is used to create classifications on the [Cifar10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html) which contains 60,000 images spread across 10 classes. These pictures are being preprocessed and then classified by applying both a logistic regression and a neutral network on the task.

## Repository Tree <a name="tree"></a>

```
├── README.md                  
├── assign_desc.md              
├── data
│   └── cifar10_preprocessed.npz    <----- pre-processed cifar data (output from preprocess.py)
├── models
│   ├── logistic_model.joblib       <----- best performing logistic regression model
│   └── neural_model.joblib         <----- best performing neural network model        
├── out
│   ├── logistic_report.txt         <----- classification report for best logistic regression model    
│   └── neural_report.txt           <----- classification report for best neural network model
├── requirements.txt            
├── run.sh
├── setup.sh
└── src
    ├── classify.py                 <----- script for running classifications
    ├── model_space.py              <----- script containing available model types + parameter grids
    ├── preprocess.py               <----- script for preprocessing Cifar10 data
    └── utils.py                    <----- script with utility functions for classifications
```


## Usage <a name="usage"></a>
This project only assumes that you have Python3 installed. If so, the analysis can be run simply by typing in the following command from terminal given you are in the root directory of the project:
```
bash run.sh
```
This bash script does the following:
* Sets up a virtual environment
* Installs requirements to that environment
* Runs `preprocess.py` (loads and preprocesses CIFAR10 dataset)
* Runs `classify.py` (classifies dataset using both LR and MLP classifier)
* Deactivates virtual environment

The logistic regression classifier and the neural network classifier are found in the `models` and their classification reports are in the `out` folder.
</br></br>

## Modified Usage
The analysis can also be run part-by-part which also allows for using additional or fewer models in the classification task. <br>

Firstly, the virtual environment and requirements are setup with the following bash script:
```
bash setup.sh
```

The data can then be processed by running the preprocessing script:
```
python src/preprocess.py
```
<br>

After having preprocessed the data, the classification can be done. In general, this analysis is setup to utilize grid search with cross-validation to optimize model parameters for the various classification types. <br>

The default is running classification is doing both logistic regression and a neural network, but they can also be run individually using the `--classifier` argument as such:
```
# run only logistic regression
python src/classify.py --classifier logistic

# run only neural network
python src/classify.py -classifier neural
```
<br>
In addition, you can easily add other classification types by simply adding them in `model_space.py` following the same format as for the current models. To demonstrate this flexibility, a RandomForest classifier is already in the model space and could be used in the analysis as such:

```
# run both logistic regression, neural network and random forest classifiers
python src/classify.py -c logistic neural random_forest
```

This would create additional outputs for RandomForest, named accordingly, and found in the `out` and `model` directories.
</br></br>

## Results <a name="results"></a>
The following results are for the best performing logistic regression and neural network classifiers within the defined parameter space used in the grid search.

### Logistic Regression Classifier
Parameters: `{'C': 0.01, 'multi_class': 'multinomial', 'penalty': 'l2', 'solver': 'saga', 'tol': 0.01}`
```
|            | Precision | Recall | F1-score | Support |
|------------|-----------|--------|----------|---------|
| airplane   | 0.35      | 0.39   | 0.37     | 1000    |
| automobile | 0.37      | 0.38   | 0.37     | 1000    |
| bird       | 0.27      | 0.21   | 0.24     | 1000    |
| cat        | 0.23      | 0.16   | 0.19     | 1000    |
| deer       | 0.25      | 0.20   | 0.23     | 1000    |
| dog        | 0.30      | 0.30   | 0.30     | 1000    |
| frog       | 0.28      | 0.33   | 0.30     | 1000    |
| horse      | 0.32      | 0.33   | 0.32     | 1000    |
| ship       | 0.34      | 0.41   | 0.37     | 1000    |
| truck      | 0.40      | 0.46   | 0.43     | 1000    |
| accuracy   |           |        | 0.32     | 10000   |
| macro avg  | 0.31      | 0.32   | 0.31     | 10000   |
| weighted avg | 0.31    | 0.32   | 0.31     | 10000   |
```
<br>

### Neural Network (MLP) Classifier
Parameters: `{'early_stopping': True, 'hidden_layer_sizes': (512, 128, 32), 'learning_rate': 'adaptive'}`
```
|            | Precision | Recall | F1-score | Support |
|------------|-----------|--------|----------|---------|
| airplane   | 0.49      | 0.47   | 0.48     | 1000    |
| automobile | 0.52      | 0.49   | 0.51     | 1000    |
| bird       | 0.38      | 0.26   | 0.31     | 1000    |
| cat        | 0.33      | 0.23   | 0.28     | 1000    |
| deer       | 0.36      | 0.34   | 0.35     | 1000    |
| dog        | 0.45      | 0.36   | 0.40     | 1000    |
| frog       | 0.46      | 0.52   | 0.49     | 1000    |
| horse      | 0.50      | 0.54   | 0.52     | 1000    |
| ship       | 0.45      | 0.62   | 0.52     | 1000    |
| truck      | 0.45      | 0.62   | 0.52     | 1000    |
| accuracy   |           |        | 0.45     | 10000   |
| macro avg  | 0.44      | 0.45   | 0.44     | 10000   |
| weighted avg | 0.44    | 0.45   | 0.44     | 10000   |
```
## Discussion <a name="discussion"></a>
The results indicate that both classifiers adapted to the task, performing above chance level. Still, as expected, the more sophisticated neural network was far superior to the logistic regression with greater F1-scores for all classes. However, it is worth noting that having used a different parameter grid could have produced different results
</br></br>
Another interesting takeaway is that both classifiers had the best accuracy for the *truck* class and struggled the most with *cat*. The poor recall scores for *cat* suggest that both models were bad at detecting a cat when presented with one in particular.
