# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 20:29:49 2020

@author: Mudit
"""

import pandas as pd

# Importing the dataset
dataset_train = pd.read_csv('image_pixels_train2.csv')
dataset_test = pd.read_csv('image_pixels_test2.csv')

X_train = dataset_train.iloc[:, 0:784].values
X_test = dataset_test.iloc[:, 0:784].values
y_train = dataset_train.iloc[:, 784].values
y_test = dataset_test.iloc[:, 784].values
# Splitting the dataset into the Training set and Test set

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 25, metric = 'minkowski', p = 2) #k=25 because greater no of classes. 25 gives better decision boundary
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
score = classifier.score(X_test, y_test)
