# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 21:45:09 2020

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

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
score = classifier.score(X_test, y_test)