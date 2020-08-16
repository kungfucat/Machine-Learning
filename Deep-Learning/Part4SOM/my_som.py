#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 22:23:50 2020

@author: harshbhardwaj
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Fraud Detection System

dataset = pd.read_csv('Credit_Card_Applications.csv')
# Frauds are usually outliers, so in case of SOM, these are the outlying neurons
# MID = Mean Inteneuron Distance

X = dataset.iloc[:, : -1].values
Y = dataset.iloc[:, -1].values
# Y represents the classes, if the customer was actually a fraud
# X will only be used in the SOM

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1) )
X = sc.fit_transform(X)

# Train the SOM
from minisom import MiniSom
# x,y represent the size of the minsom grid
# input_len = number of features
# sigma = radius of the different neighbours in the grid
som = MiniSom (x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)


# Visualising the Results
# Higher MID means more the winning node is an outlier
from pylab import bone, pcolor, colorbar, plot, show

# Empty window
bone()
# Distance_Map function returns the All the MID in one matrix, for all the winning nodes
# Need transpose of this for pcolor function
pcolor(som.distance_map().T)
# Will add a legend to find what white represents and what black represents
colorbar()

# ------------------------------------------------------------------------------------------------- #
# Could have stopped here
# Red circles = didn't get approval
# Green Circles = did get approvals
markers = ['o', 's']
colors = ['r', 'g']

# x represents the row vector for every i
# w[0] and w[1] represents the coordinates of the lower left side corner
# we want to draw/plot the figures at the centre, so we add 0.5
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

# Finding the frauds
mappings = som.win_map(X)
# (6,8) and (8,1) were empty, leading to issues in concatenation
# Can handle that seperately, just dummy code below
frauds = np.concatenate( (mappings[(0, 0)], mappings[(1,1)]), axis = 0)
frauds = sc.inverse_transform(frauds)

print('Fraud Customer IDs')
for i in frauds[:, 0]:
  print(int(i))