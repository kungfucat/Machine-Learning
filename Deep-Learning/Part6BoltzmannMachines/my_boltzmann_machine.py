#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 20:03:29 2020

@author: harshbhardwaj
"""

# Basically making two recommender systems, i.e. if a user is going to like a movie, yes or no [Using Restricted Boltzmann Machines]
# And one system to recommend the rating of the movie by the user [in Auto Encoders]

# Boltzmann Machines

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
# Movie names may contain commas, so the seperator used in ::
# encoding: Some special symbols cannot be encoded using UTF-8, so we specify it
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Col. 0 = ID, Col. 1 = Gender, Col. 2 = age, Col. 3 = Codes corresponding to User's Job, Col. 4 = ZIP Code
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Col. 0 = ID of user, Col. 1 = Movie IDs, Col. 2 = Ratings[1 to 5], Col. 3 = Timestamps when rated(won't use this)
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing Training and Test Set
# u1 base and u1 test etc. are 5 sets for the same 1 million ratings for K-cross validation
# We will use only one pair of train and test sets
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
# For pytorch we need arrays and not data frames, so will convert
training_set = np.array(training_set, dtype = 'int')

test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting total number of users & movies
nb_users = int(max(max(training_set[:, 0], ), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1], ), max(test_set[:, 1])))

# For both training & test sets, we will make a matrix with users as rows and movies as columns
# And for user U and movie M, arr[U][M]  will have the rating by the user U for movie M
# If U didn't watch the movie, we will put a 0 simply
# Why: develop a structure of data which a restricted boltzmann machine can accept

# Converting the data into an array with users in lines and movies in columns
def convert(data):
	new_data=[]
	for id_users in range(1, nb_users + 1):
		# select all id_movies for current user, the condition is written in square brackets 
		id_movies = data[:, 1][data[:, 0] == id_users]
		id_ratings = data[:, 2][data[:, 0] == id_users]
		# list of all the movies for current user
		ratings = np.zeros(nb_movies)
		# -1 as movie with id 1, will be at index 0 and so on
		# Python automatically does one on one matching here
		ratings[id_movies -  1] = id_ratings
		new_data.append(list(ratings))
	return new_data

training_set = convert(training_set)    
test_set = convert(test_set)

# Convert data into torch tensors
# Tensors = Arrays which contain elements of a single type
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Convert ratings into binary ratings liked(1) and disliked(0)
# Why? : Because we won't actually predict ratings here, just recommend movies
training_set[training_set == 0] = -1 # Did not see the movie
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# Creating Architecture of the network
class RBM():
	# nv = number of visible nodes
	# nh = number of hidden nodes
	def __init__(self, nv, nh):
		
		# Weights,  Probabilities of the visible nodes given the hidden nodes
		# Matrix of nh and nv
		# randn = random weights based on normal distribution, mean=0 variance=1
		self.W = torch.randn(nh, nv)
		
		# Bias of the probabilities of the hidden nodes given the visible nodes(PH given V)
		# First dimensions corresponds to batch and second is for the tensor
		# So a 2d matrix, not a 1d array 
		self.a = torch.randn(1, nh)
		# Bias for the visible nodes
		self.b = torch.randn(1, nv)
		
	def sample_h(self, x):
		# To sample the hidden nodes
		# x corresponds to visible neurons V in the probabilities PH given V
		# sampling the hidden nodes according to the probabilities PH given V which actually is the sigmoid activation function
		# Why we need this: Because during the training, we will approximate the log likelihood gradient using Gibbs Sampling
		# In simple words if there are 100 hidden nodes, this function will activate them according to certain probability, which
		# we compute here, i.e. p(h=1 | V)
		# PH given V = sigmoid function applied to W*X+a(bias), in paper
		wx = torch.mm(x, self.W.t())
		# WX is in a 'minbatch'
		# So we need to make sure this bias is applied to each dimension of this minibatch
		# So we use a function which adds a new dimension for the bias we are adding
		activation = wx + self.a.expand_as(wx)
		
		# p_h_given_v = probability that hidden node h is activated given v
		p_h_given_v = torch.sigmoid(activation)
		
		
		# We are making a Bernouilli RBM as we are predicting only a binary outcome
		# If ph given V is 0.7, then to decide if the hidden node is activated or not
		# We use the Bernouilli distribution to generate a random number between 0 and 1 and if
		# it is greater than phGivenV, then it will return 1(hidden node activated) or else 0 
		return p_h_given_v, torch.bernoulli(p_h_given_v)
	
	def sample_v(self, y):
		# To sample visible nodes, as we need to return a 0 or 1    
		wy = torch.mm(y, self.W)
		activation = wy + self.b.expand_as(wy)
		p_v_given_h = torch.sigmoid(activation)
		return p_v_given_h, torch.bernoulli(p_v_given_h)
	
	def train(self, v0, vk, ph0, phk):
		# v0 = input vector
		# vk = visible nodes after k samplings(from V to H and back)
		# ph0 = vector of probabilities at 0th iteration
		# phk = vector of probabilities at kth iteration
		
		# Contrastive Divergence to approximate the log likelihood gradient 
		# Why gradient? as we need to maximise the log likelihood for the training set
		# Why approximate? Because calculations will be too complex otherwise
		# We use Gibbs Sampling for this
		self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
		# torch.sum is to maintain 2 dimensions of self.b
		self.b += torch.sum((v0 - vk), 0)
		self.a += torch.sum((ph0 - phk), 0)
		
		
# nv = number of movies		
nv = len(training_set[0])
# We chose this number
nh = 100
# Update weights after 'batch_size' train samples
batch_size = 100
rbm = RBM(nv, nh)

# Training the RBM
nb_epoch = 10

for epoch in range(1, nb_epoch+1):
	# to measure distance between predicted and calculated values
	train_loss = 0
	# Nomalise the train loss
	s = 0.0
	for id_user in range(0, nb_users - batch_size, batch_size):
		# Input = ratings of all movies by this user, i.e. id_user
		# Target in the beginning is the same as the input
		# input will keep changing, target remains same
		
		# output of Gibbs sampling, but initially is the ratings which already existed
		# target = v0, predicted = vk
		vk = training_set[id_user : id_user + batch_size]
		v0 = training_set[id_user : id_user + batch_size]
		ph0, _ = rbm.sample_h(v0)
		
		# Looping for the k steps of contranstive Divergence
		# i.e. from visible to hidden nodes and back
		for k in range(10):
			_, hk = rbm.sample_h(vk)
			_, vk = rbm.sample_v(hk)
			# Not learn from cells with -1 in them, so we will freeze them this way:
			vk[v0 < 0] = v0[v0 < 0]
		phk, _ = rbm.sample_h(vk)
		rbm.train(v0, vk, ph0, phk)
		# calculate loss only for those nodes having existent ratings
		train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
		s += 1.0
	train_loss = float(train_loss) 
	print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
	
# Testing the RBM
test_loss = 0
s = 0.0
# Looping over every user in test set
for id_user in range(nb_users):
	# v = input to make predictions on
	# Will have training set here, because it is the input to activate the neurons to predict rating of the movies not rated yet
	v = training_set[id_user : id_user + 1]
	# vt = target
	# test_set as we will compare this with the above variable
	vt = test_set[id_user : id_user + 1]
	# If we have something data for the test subject, then only make predictions
	if len(vt[vt>=0]) > 0:
		_, h = rbm.sample_h(v)
		_, v = rbm.sample_v(h)
		test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
		test_loss=float(test_loss)
		s += 1.0
print('test loss: '+str(test_loss/s))
	