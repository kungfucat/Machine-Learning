# used in computer vision and medicine, high power consuming tasks,
# can be done easily, which is the backbone of deep learning
# Recognising objects, or tumers in brain even
# Recommender systems using deep boltzmann machines

# Theano = numerical computation library, based on numpy syntax
# and not only run on CPU(main processor, general task) but also on GPU(processor for graphic, lot more fractional
# operations per second kar sakta hai),
# GPU is much better for deep learning models

# Tensorflow = numeric computation library, very fast
# Under Apache 2.0 license
# Mostly for research and development

# Keras- kind of wraps the rest of the 2 library, very few lines of code


# THIS IS A GEO DEMOGRAPHIC MODEL
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# Column 1,2,3 have no impact on the decision whether to leave or not
dataset = pd.read_csv('Churn_Modelling.csv')
# 13 excluded, 12 tak lega
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()
# It is done cause ML dont work on strings, so converting each column to numbers
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
# one hot encoder(dummy encoding)

# Male-Female
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()

# Will remove one dummy varaible to avoid falling into
#  the dummy variable trap
X = X[:, 1:]

# Splitting the data set into training and testing set
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling, We need it, Compulsory kind of
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# MAKE ANN:

# Importing Keras libraries and packages
# Sequential module to initialise our ANNmodel
# Dense module to build layers of our ANN model
import keras
from keras.models import Sequential
from keras.layers import Dense

# Two ways to initialise, by defining the layer
# or by defining the graph
# We will use sequence of layers
# ANN acting as a classifier
classifier = Sequential()
# Will add sequence of layers, layer by layer
# Dense function will take care of step 1 from slide
# Each input node is the independent variable
# Using sigmoid function at output level will give probability
# Which will be pretty cool
# By how much weight should be updated is determind by the learning factor

# ADDING INPUT LAYER AND FIRST HIDDEN LAYER
# No. of nodes of hidden layer=output_dim, art actually
# If data is linearly seperable, no layer may be needed
# Average of nodes in input and output layer is a good start
# But better use parameter tuning for using art
# Input mein 11 node, output mein 1 node
# So average pe 6, init is for initial weights of the model
# In hidden layer, so specified activation function as rectifier
# rectifier ka naam is relu
# No. of input nodes bhi daalni hai, i.e 11
# We won't need input_dim for the other hidden layers,
# Next hidden layer knows what to expect

# init = uniform ka matlab ki randomly assign weights,
# with values closer to 0
# Adding input and first hidden layer
classifier.add(Dense(output_dim=6,
                     init='uniform',
                     activation='relu',
                     input_dim=11))

# LATEST API VERSION mein use : Dense(input_dim=11, kernel_initializer="uniform", activation="relu", units=6)

# Adding second hidden layer, we don't need input_dim here
# because it knows what to expect, we will keep it 6 again
classifier.add(Dense(output_dim=6,
                     init='uniform',
                     activation='relu'))

# Adding the output layer
# Output mein 1 hi node chahiye, uniform hi rakhenge for weights
# coming to the output layer
# soft-max when used in activation,
# actually sigmoid function hi hai, used when applied
# to a DV with more than 2 categories
classifier.add(Dense(output_dim=1,
                     init='uniform',
                     activation='sigmoid'))

# DONE ADDING THE LAYERS
# Compiling the ANN, means applying the Stochastic gradient descendent on the whole network

# optimiser is the algorithm to find the optimal weights,
# bhot saare algorithms hote hain Stochastic gradient mein, we use adam, which is very efficient
# loss function within the adam algorithm
# loss function jaise linear regression mein distance b/w actual and expected value ko minimise kara tha, same waise hi

# Will use logarithmic loss function
# categorical_crossentropy naame hai when more than 2 categories
# warna called binary_cross entropy
# metrics, criterion to evaluate the model, accuracy used
# uses this criterion to optimise weights
# metrics wants a list and single element metric with accuracy is used
classifier.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

# Fitting the ANN to training set
# batch size:no. of observations after which to update weights
# epochs : a round after which whole dataset passed through ANN
classifier.fit(x=X_train, y=Y_train, batch_size=10,
               epochs=100)

# Predict the result
y_pred = classifier.predict(X_test)

# choose threshold when predicted result is 1 or not
#if y_pred >0.5, returns 1, else return 0
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, y_pred)
