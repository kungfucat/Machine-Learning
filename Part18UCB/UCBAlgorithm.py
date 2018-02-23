# Upper Confidence Bound


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# for sqrt
import math

# ctr=clickThroughRate
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
# 10 different ads, so which ad to put on social network
# 0 and 1 means, 0 means notClicked, 1 means clicked
# 10k users, strategy depends on the results observed previously
# strategy is dynamic in nature
# ith user can click on the jth column ad
# only when arr[i][j]=1
# When randomly choosing ads, we got 1200 reward on an average

# IMPLEMENTING SAME AS IN SLIDE
# Implementing UCB
# n is total rounds
# d is number of ads
N = 10000
d = 10
# this creates a vector of size d with all values 0
# Ni(n) hai ye
numbersOfSelections = [0] * d
# Ri(n) hai ye
sumsOfRewards = [0] * d
# totalReward for each round
totalReward = 0
# Huge vector of all the different versions of ads selected at each step
# ad selected at each stage
# FOR THE FIRST 10 ads, we simply pick the 10 first ads
# at round 1 choose ad 1,round 2 choose ad 2 and so on
adsSelected = []
for n in range(0, N):
    ad = 0
    # will give the maxUpperBound for each row
    maxUpperBound = 0
    for i in range(0, d):
        # never true when selecting ad for the first time
        if numbersOfSelections[i] > 0:
            averageReward = sumsOfRewards[i] / numbersOfSelections[i]
            # this formula needs start from 1, so we did +1 to log
            deltaI = math.sqrt(3 / 2 * math.log(n + 1) / numbersOfSelections[i])
            upperBound = averageReward + deltaI
        else:
            # 10^400 why?
            # When we choose an ad i for the first time
            # so we use this, and maxUpperBound=10^400
            # for ad 1, 10^400 is not > 10^400, so ad=0 lenge
            # for ad=1, ad = 0 ke liye if true hoga, and
            # 1 ke liye 10^400 hi lenge
            # set Such a high value ki upar jo upperBound aaye
            # Ussey bada aaye for ad i for the first time
            upperBound = 1e400
        # Caclulate the MaxUpperBound
        if upperBound > maxUpperBound:
            maxUpperBound = upperBound
            # Ad selected
            ad = i
    # ad will be the ad selected
    adsSelected.append(ad)
    # number of times ad was selected increased by 1,
    #  so add 1
    numbersOfSelections[ad] = numbersOfSelections[ad] + 1
    # nth row ke liye adth column selected, so uski value,
    # add it to sumOfRewards for that row
    reward = dataset.values[n, ad]
    sumsOfRewards[ad] = sumsOfRewards[ad] + reward
    totalReward = totalReward + reward

# Total reward 2178 aaya
plt.hist(adsSelected)
plt.title('Histogram of ads selection')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad selected')
plt.show()
# ad 5(index 4 wali should be chosen)
