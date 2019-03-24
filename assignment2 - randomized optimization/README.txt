## Overview

mlrose is a Python package for applying some of the most common randomized optimization and search algorithms to a range of different optimization problems, over both discrete- and continuous-valued parameter spaces. This notebook contains the examples used in the mlrose tutorial.

### Import Libraries

import mlrose
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

import time, datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import xgboost as xgb

from sklearn.model_selection import cross_val_predict, cross_val_score, GridSearchCV, KFold, GroupKFold, GroupShuffleSplit, learning_curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import explained_variance_score, make_scorer

# fix random seed for reproducibility
np.random.seed(121)
%matplotlib inline

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer

data_cancer = load_breast_cancer()
print(data_cancer.keys())
x2, y2 = pd.DataFrame(data_cancer["data"],columns=data_cancer.feature_names), pd.Series(data_cancer["target"])

# shuffle the indices first
n2 =np.random.RandomState(seed=121).permutation(range(len(x2)))

# Split indices
t2 ,v2 , r2 = x2.iloc[-114:,:].index, x2.iloc[-228:-114,:].index, x2.iloc[:-228,:].index
print(t2, v2, r2)

y2.hist(bins=10)

# dataset splits
x2_ts, x2_val, x2_tr = x2.iloc[t2,:], x2.iloc[v2,:], x2.iloc[r2,:]
y2_ts, y2_val, y2_tr = y2[t2], y2[v2], y2[r2]

x2_ = x2.iloc[list(r2)+list(v2),:]
y2_ = y2[list(r2)+ list(v2)]
print(len(x2_), len(y2_))

def plot_curve(c, dataset):
    # instantiate
    clf = c[1]
    # fit
    clf.fit(X, y)
      
    train_sizes, train_scores, test_scores = learning_curve(clf, X, y, n_jobs=-1, cv=cv, scoring="accuracy",
                                                            train_sizes=np.linspace(.1, 1.0, 5), verbose=0)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.figure()
    plt.title(c[0]+ " on Dataset: "+ dataset)
    plt.legend(loc="best")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.gca().invert_yaxis()
    
    # box-like grid
    plt.grid()
    
    # plot the std deviation as a transparent range at each training set size
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
    # plot the average training and test score lines at each training set size
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    
    # sizes the window for readability and displays the plot
    # shows error from 0 to 1.1
    plt.ylim(-.1,1.1)
    plt.show()

def set_data(dataset):
    if dataset=="digits":
        dataset="digits"
        X, y =x1_, y1_
    else:
        dataset="diagnosis"
        X, y =x2_, y2_
    return X, y, dataset

X, y, dataset = set_data("")
results=[]
for i in range(1,200,20):
    clf = MLPClassifier(random_state=121, hidden_layer_sizes=(i,))
    # Perform 5-fold cross validation 
    scores = cross_val_score(estimator=clf, X=X, y=y, cv=5, n_jobs=-1)
    results.append(scores.mean())
#print(results)

plt.plot(range(1,200,20), results)
plt.title("MLPClassifier on "+ dataset)
plt.ylim(0,1)
plt.xlabel("Number of Neurons in Hidden Layer")
plt.ylabel("Accuracy Score")
;

# Take 75 as the number of neurons in middle layer

# Hidden Layer Sizes
X, y, dataset = set_data("")
results=[]
for i in range(1,40,2):
    clf = MLPClassifier(random_state=121, hidden_layer_sizes=75, max_iter=i ) 
    # Perform 3-fold cross validation 
    scores = cross_val_score(estimator=clf, X=X, y=y, cv=3, n_jobs=-1)
    results.append(scores.mean())
#print(results)

plt.plot(range(1,40,2), results)
plt.title("MLPClassifier on "+ dataset)
plt.ylim(0,1)
plt.xlabel("Number of Iterations")
plt.ylabel("Accuracy Score")
;

### Example 6: Fitting a Neural Network to the Iris Dataset

# Load the Iris dataset
data = load_breast_cancer()
# Get feature values of first observation
print("\n", data.data[0])
# Get feature names
print("\n",data.feature_names)
# Get target value of first observation
print("\n",data.target[0])
# Get target name of first observation
print("\n",data.target_names[data.target[0]])
# Get minimum feature values
print("\n",np.min(data.data, axis = 0))
# Get maximum feature values
print("\n",np.max(data.data, axis = 0))
# Get unique target values
print("\n",np.unique(data.target))

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size = 0.2, 
                                                    random_state = 3)

# Normalize feature data
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# One hot encode target values
one_hot = OneHotEncoder()

y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()

# make function to iterate over the methods
# vary the number of iterations 200, 500, 1000, 2000, 5000, 10000, 20000
# vary the number of hidden layers: 1,2,3,4,5

iterator = [200, 500, 1000, 2000, 5000, 10000]
hidden = [[x] for x in range(1,50,10)]
algs = ["gradient_descent", "random_hill_climb", "simulated_annealing", "genetic_alg"]

def randomized_search(alg, hnodes, iters):
    # Initialize neural network object and fit object - attempt 1
    np.random.seed(3)
    nn_model1 = mlrose.NeuralNetwork(hidden_nodes = hnodes, activation ='relu', 
                                     algorithm = alg, 
                                     max_iters = iters, bias = True, is_classifier = True, 
                                     learning_rate = 0.0001, early_stopping = True, 
                                     clip_max = 5, max_attempts = 100)
    nn_model1.fit(X_train_scaled, y_train_hot)

    # Predict labels for train set and assess accuracy
    y_train_pred = nn_model1.predict(X_train_scaled)
    y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)

    # Predict labels for test set and assess accuracy
    y_test_pred = nn_model1.predict(X_test_scaled)
    y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)

    #print("y_train_accuracy : ", y_train_accuracy, "y_test_accuracy : ", y_test_accuracy)
    return y_train_accuracy, y_test_accuracy

randomized_search(algs[0], hidden[0], iterator[0])

# vary the hidden units

algs[0]

alg

#for alg in algs:
alg = algs[0]    
x_vals, y_vals = hidden, []
for h in hidden:
    #print(randomized_search(alg, h, 1000)[1])
    y_vals.append(randomized_search(alg, h, 1000)[1])

plt.plot(x_vals, y_vals)    
plt.title("Gradient Descent - Number of hidden layer units vs. Accuracy")
plt.ylim(0.9, 1)

# vary the number of iterations

x_vals1, y_vals1 = iterator, []
for i in iterator:
    y_vals1.append(randomized_search(alg, [10], i)[1])

plt.plot(x_vals1, y_vals1)
plt.title("Gradient Descent -  Number of Iterations vs. Accuracy")
plt.ylim(0.9, 1)

#10, 2000

# choose the best for all algs and measure the time at standard parameters

#best hidden units = 10
#best number of iterations = 1000

times = []
acc = []
for alg in algs:
    t1 = time.time()
    calc = randomized_search(alg, [10], 1000)[1]
    t2 = time.time()
    acc.append(calc)
    times.append(t2-t1)    

times

70/2.9

acc

df = pd.DataFrame([ algs, acc, times]).T.rename(columns={0:"algorithm", 1: "accuracy", 2:"time_seconds"})

d = df.sort_values(by="accuracy", ascending=False)
d.set_index("algorithm", inplace=True)
d["accuracy"].plot(kind="barh")
plt.title("Comparison of Randomized Optimization Methods - Performance")

d = df.sort_values(by="time_seconds", ascending=False)
d.set_index("algorithm", inplace=True)
d["time_seconds"].plot(kind="barh")
plt.title("Comparison of Randomized Optimization Methods - Time in seconds")

d.sort_values(by="accuracy")

algs

# RHC

#for alg in algs:
alg = algs[1]    
x_vals, y_vals = hidden, []
for h in hidden:
    #print(randomized_search(alg, h, 1000)[1])
    y_vals.append(randomized_search(alg, h, 1000)[1])

plt.plot([1,11,21,31,41], y_vals)    
plt.title("RHC - Number of hidden layer units vs. Accuracy")
plt.ylim(0, 1)

# vary the number of iterations
x_vals1, y_vals1 = iterator, []
for i in iterator:
    y_vals1.append(randomized_search(alg, [11], i)[1])

plt.plot(x_vals1, y_vals1)
plt.title("RHC -  Number of Iterations vs. Accuracy")
plt.ylim(0, 1)

# SA

#for alg in algs:
alg = algs[2]    
x_vals, y_vals = hidden, []
for h in hidden:
    #print(randomized_search(alg, h, 1000)[1])
    y_vals.append(randomized_search(alg, h, 1000)[1])

plt.plot([1,11,21,31,41], y_vals)    
plt.title("SA - Number of hidden layer units vs. Accuracy")
plt.ylim(0, 1)

# vary the number of iterations
x_vals1, y_vals1 = iterator, []
for i in iterator:
    y_vals1.append(randomized_search(alg, [11], i)[1])

plt.plot(x_vals1, y_vals1)
plt.title("SA -  Number of Iterations vs. Accuracy")
plt.ylim(0, 1)

# GA

"""#for alg in algs:
alg = algs[3]    
x_vals, y_vals = hidden, []
for h in hidden:
    #print(randomized_search(alg, h, 1000)[1])
    y_vals.append(randomized_search(alg, h, 1000)[1])

plt.plot([1,11,21,31,41], y_vals)    
plt.title("GA - Number of hidden layer units vs. Accuracy")
plt.ylim(0, 1)""";

# vary the number of iterations
x_vals1, y_vals1 = iterator, []
for i in iterator:
    y_vals1.append(randomized_search(alg, [11], i)[1])

plt.plot(x_vals1, y_vals1)
plt.title("GA -  Number of Iterations vs. Accuracy")
plt.ylim(0, 1)

#10, 2000

# choose the best for all algs and measure the time at standard parameters

#best hidden units = 10
#best number of iterations = 1000

times = []
acc = []
for alg in algs:
    t1 = time.time()
    calc = randomized_search(alg, [10], 1000)[1]
    t2 = time.time()
    acc.append(calc)
    times.append(t2-t1)    

times

acc

df = pd.DataFrame([ algs, acc, times]).T.rename(columns={0:"algorithm", 1: "accuracy", 2:"time_seconds"})

d = df.sort_values(by="accuracy", ascending=False)
d.set_index("algorithm", inplace=True)
d["accuracy"].plot(kind="barh")
plt.title("Comparison of Randomized Optimization Methods - Performance")

d = df.sort_values(by="time_seconds", ascending=False)
d.set_index("algorithm", inplace=True)
d["time_seconds"].plot(kind="barh")
plt.title("Comparison of Randomized Optimization Methods - Time in seconds")

d.sort_values(by="accuracy")