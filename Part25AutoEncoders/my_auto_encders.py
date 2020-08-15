#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 19:15:11 2020

@author: harshbhardwaj
"""

# Importing the Libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training and test sets
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')


# Maximum number of users and movies
nb_users = int(max(max(training_set[:, 0], ), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1], ), max(test_set[:, 1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
	new_data=[]
	for id_users in range(1, nb_users + 1):
		id_movies = data[:, 1][data[:, 0] == id_users]
		id_ratings = data[:, 2][data[:, 0] == id_users]
		ratings = np.zeros(nb_movies)
		ratings[id_movies - 1] = id_ratings
		new_data.append(list(ratings))
	return new_data

training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into torch sensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Creating the architecture of the neural network
# SAE = Stacked Auto Encoders, several hidden layers
class SAE(nn.Module):
	def __init__(self, ):
		super(SAE, self).__init__()
		# connection between input and the first hidden layer
		# fc = full connection
		# Encoding
		self.fc1 = nn.Linear(nb_movies, 20)
		self.fc2 = nn.Linear(20, 10)
		# Decoding
		self.fc3 = nn.Linear(10, 20)
		self.fc4 = nn.Linear(20, nb_movies)
		self.activation = nn.Sigmoid()
		
	def forward(self, x):
		# x is the input vector
		x = self.activation(self.fc1(x))
		x = self.activation(self.fc2(x))
		x = self.activation(self.fc3(x))
		# No need of applying the activation function on the last output node
		x = self.fc4(x)
		return x
	
sae = SAE()
#criterion for the loss function
criterion = nn.MSELoss()
# decay is used to reduce the learning rate with time
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)


# Training the SAE

nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
	train_loss = 0
	# Users who rated at least one movie and we completely give up on users who didn't rate any movie
	s = 0.
	for id_user in range(nb_users):
		# raining_set[id_user] is a vector, but we want a batch of vectors
		# So we add another dimension at index 0
		input = Variable(training_set[id_user]).unsqueeze(0)
		target = input.clone()
		# This condition is to only to save memory i.e. optimise the code
		if torch.sum(target.data > 0) > 0:
			# Forward function is applied automatically
			output = sae(input)
			# This ensures we don't calculate gradient w.r.t to the target variable
			# Saves computations, optimises code
			target.require_grad = False
			# From output, take ratings such as target = 0
			# This won't count in the computations
			# Optimises code
			output[target == 0] = 0
			loss = criterion(output, target)
			# number of movies divided by movies having positive rating
			# This is just to adapt to the condition we stated earlier that is, computing where rating is non zero
			mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
			# Just to tell in which direction we need to update the weights
			loss.backward()
			train_loss += np.sqrt(loss.data*mean_corrector)
			s += 1.
			# Amount by which updates of weights should be done, loss just mentions the direction
			optimizer.step()
	print('epoch: '+str(epoch)+'loss: '+ str(train_loss/s))
	
# Testing the SAE

test_loss = 0
s = 0.
for id_user in range(nb_users):
	input = Variable(training_set[id_user]).unsqueeze(0)
	target = Variable(test_set[id_user]).unsqueeze(0)
	if torch.sum(target.data > 0) > 0:
		output = sae(input)
		target.require_grad = False
		output[target == 0] = 0
		loss = criterion(output, target)
		mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
		test_loss += np.sqrt(loss.data*mean_corrector)
		s += 1.
		
print('test loss: '+str(test_loss/s))