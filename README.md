## Description
This is a relatively small exercise in developing an ML application using Python.

Application reads provided comma separated files `training_data_example.csv`, `validation_data_example.csv` and `employee.csv` are included in this repo, for the sake of example, in order to address the common problem of mixing of personal and business expenses in a small business.  The algorithm is intended to separate any potential personal expenses in the training data.  Labels of personal and business expenses were deliberately not given as this is often the case in many systems.

The objective therefore is to train a learning model that assigns each expense transaction to one of the set of predefined categories and evaluate it against the validation data provided.  The set of categories are those found in the "category" column in the training data, and report on accuracy and at least one other performance metric.

## Pre-requisites
1. Python on system with environmental variables set (I use Python 2.7).
2. Some none-standard libraries might need to be installed if not already there. The main ones are the Scikit-Learn and the IPython 	    distributed computing libraries .
3. To make things easier there are no file path settings to follow, just ensure that all files are in the same path/folder. However, for    Q3, make sure that copies of the source csv files you provided in the IPython engine default local path at
   `/Python27/Lib/site-packages/ipyparallel/`
4. I use PyCharm IDE for working with Python, so a .idea folder is there if needed.


## How-to
There are 4 .py files independent of each other, so you may run them separately in any order you wish: 
1. `main.py` is the code that implements the main task.
2. `feature_selection.py` is an exercise that shows why I chose the features to use.
3. `expense_type_algo.py` is a possible solution for this ML problem, and.It uses a variant of the csv files  provided with some          	    additional info I put in manually based on some assumptions.
4. `mainWcluster.py` is a simple example of distributed computing using IPython.

## What algorithms and why

The candidate algorithms for this problem were `Naive Bayes(Gaussian)`, `Gradient Boost`, `SVC SVM`, `Linear Regression`, `Logisitic Regression` and `Linear Discriminant Analysis`.` Cross validation` was then conducted to see which would give higher accuracy using the validation set, and the one with highest score is then tested as per question instructions. These were selected because in practice I find them to be the best to deal with a sample of few instances and features, as more complex algorithms and ensembles tend to overfit and bias in such cases (Although realistically how a library or code is actually implementing an algorithm and on what platform tend to factor heavily).

For this problem I used an NLP(ish) text tokenizing approach to create classification features. I've always wanted to try this type of mining, and this seemed like a good example to try because the only real lead in the training/validatng csv files for differentiating between expenses was the expense description field, and as it's a 'free form' text field a text feature extraction based algorithm seemed a good option. For simplicity's sake and allowing for practical implementation concerns, the classification is split into `business` and needing `review` for reasons again I will explain if we get to talk.
A promising possible supplement to this algorithm is to factor in employees(i.e. their IDs) in the features as well, since the it is a sensible supposition that certain employees would incur higher business expenses than others (e.g. sales team would have more travel expenses). 

## Overall algorithm performance

Running the code produces a metrics matrix csv file detailing performance. Prediction is around 89 percent accurate using `Gradient Boost`, however `SVM SVC` achieves similar results and would be more efficient and significantly faster . Out of curiousity I applied a `Random Forest` ensemble and managed to achieve over 97 percent accuracy, but again the dataset is just much too small for more complex algorithms.

## Currently working on

More data in order to evaluate necessary preprocessing (Normalization Vs. Standardization, what dimension reduction is needed, etc.) in a real-life implementation.

Testing with the larger dataset in a distributed environment (The intention is to use SPARK). Code and data connection additions/modifications would be needed accordingly.
