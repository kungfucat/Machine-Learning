#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 10:16:37 2020

@author: harshbhardwaj
"""

# ANN + SOM
# Ranking customers based on the probability that ith customer is a fraud

# Part 1: SOM - Self Organising Maps

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Credit_Card_Applications.csv')

X = dataset.iloc[:, : -1].values
Y = dataset.iloc[:, -1].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1) )
X = sc.fit_transform(X)

from minisom import MiniSom
som = MiniSom (x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

from pylab import bone, pcolor, colorbar, plot, show

bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']

for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5, 
         w[1] + 0.5,
         markers[Y[i]],
         markeredgecolor = colors[Y[i]],
         markerfacecolor = None,
         markersize = 7,
         markeredgewidth =  1)

show()

mappings = som.win_map(X)
frauds = np.concatenate( (mappings[(8, 3)], mappings[(6,1)]), axis = 0)
frauds = sc.inverse_transform(frauds)

print('Fraud Customer IDs')
for i in frauds[:, 0]:
    print(int(i))



# Part 2: Going from unsupervised to Supervised Deep Learing

# Creating matrix of features
# Remove customer ID, and include the dependent variable
customers = dataset.iloc[:, 1 :].values


# Creating Dependent Variable using SOM
is_fraud = np.zeros(len(dataset))

# No .values as it is necessary to create numpy array only
for i in range(len(dataset)):
    if(dataset.iloc[i, 0] in frauds):
        is_fraud[i] = 1


        
# Part 3: ANN - Artificial Neural Network

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense (units = 2, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15))
classifier.add(Dense (units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

"""## Training the ANN on the Training set"""
classifier.fit(customers, is_fraud, batch_size = 1, epochs = 5)

"""## Predicting Probabilities of fraud"""
y_pred = classifier.predict(customers) 
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis = 1)

# 1 is to sort the 2d array on the basis of the probability 
y_pred = y_pred[ y_pred[:, 1].argsort()]