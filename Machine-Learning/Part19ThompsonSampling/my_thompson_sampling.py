#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 13:08:32 2020

@author: harshbhardwaj
"""

# Thompson Sampling

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')


# Implementing Thompson Sampling
import random
N = 10000
d = 10
ads_selected = []
# number of 1s for ad i, upto round n
numbers_of_rewards_1 = [0] * d
# number of times each ad got a 0 upto round n
numbers_of_rewards_0 = [0] * d
total_reward = 0

for n in range(N):
	ad = 0
	max_random = 0
	for i in range(0, d):
		random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
		if(random_beta > max_random):
			max_random = random_beta
			ad = i
	ads_selected.append(ad)
	reward = dataset.values[n, ad]
	if(reward == 1):
		numbers_of_rewards_1[ad] += 1
	else:
		numbers_of_rewards_0[ad] += 1
	total_reward += reward
	
	
print("Total Reward:" + str(total_reward))
# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()