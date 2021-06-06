# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 00:03:07 2020

@author: SR
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv("Admission_Predict.csv")
x = dataset.iloc[:, [1, 2]].values
y = dataset.iloc[:, 7].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
classifier  = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p=2)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

cm= confusion_matrix(y_test, y_pred)
report= classification_report (y_test, y_pred)
print(cm)
print(report)