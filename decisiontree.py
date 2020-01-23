# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 19:28:07 2020

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
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)"""

"""from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)"""

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
score = classifier.score(X_test, y_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)