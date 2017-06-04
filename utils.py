import numpy as np
import pandas as pd
import sys
import os
from sklearn.linear_model import SGDClassifier

data_path = "../Data/"
os.getcwd()
os.chdir(data_path)

def logloss(y, pred):

    epsilon = 1e-15
    pred = np.maximum(epsilon, pred)
    pred = np.minimum(1-epsilon, pred)
    l1 = np.sum(y * np.log(pred) + (1 - y) * np.log((1 - pred)))
    return l1 * -1.0/len(y)

# model-lr

def logisticRegression(X_train, y_train):
    clf = SGDClassifier(loss="log", penalty="l2", n_iter=10)
    clf.fit(X_train,y_train)
    return clf

# model-gbdt

