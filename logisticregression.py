# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 22:05:15 2020

@author: Mudit
"""

import pandas as pd

# Importing the dataset
dataset_train = pd.read_csv('image_pixels_train.csv')
dataset_test = pd.read_csv('image_pixels_test.csv')

X_train = dataset_train.iloc[:, 0:784].values
X_test = dataset_test.iloc[:, 0:784].values
y_train = dataset_train.iloc[:, 784].values
y_test = dataset_test.iloc[:, 784].values

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, max_iter=50000, multi_class = 'multinomial')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
score = classifier.score(X_test, y_test)