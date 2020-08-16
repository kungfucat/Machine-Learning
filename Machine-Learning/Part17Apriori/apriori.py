# Apriori Association Rule Mining
# Instead of using any package, we will use
# Apyori file : implementation of Apriori
# from the PythonSoftwareFoundation

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
# X axis shows the name products,
# No headers in the dataset, so did header=None
# Each observation line shows a transaction by a user
# What apyori expects is a list of list
# So will prepare input first
# Created an empty vector, whole list of transactions
transactions = []
# Upper bound not included, so chose sizeOfDataset+1
for i in range(0, 7501):
    # Apyori wants strings, so str() before dataset
    # this loop updates j accordingly and travels the row
    # accodingly
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

# Training Apriori on the dataset
# apriori function imported from apyori
from apyori import apriori

# Second argument are keyword arguments,
# depending on our dataset
# length is number of transactions we want
# once set, and if the rules don't make sense, change these params
# the min values vary from model to model
# support = product purchased 3 times a day(average), i.e 21 times a week
# divide by total transaction to get support
# confidence too high(say 80%)
# will result is products which are sold the most
# because they will only fulfil the high confidence value
# so chose 20%
rules = apriori(transactions=transactions,
                 min_support=0.003,
                min_confidence=0.2,
                min_lift=3,
                min_length=2)

# Visualise the results
# rules by apriori are all sorted by their relevance
# relevance is support+confidence+lift and not just lift
results = list(rules)
