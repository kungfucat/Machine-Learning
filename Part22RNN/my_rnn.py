#!/usr/bin/env python3

"""
Created on Sat Aug  1 18:24:44 2020

@author: harshbhardwaj

"""

# To predict stock price trend of google stocks
# According to Brownian Motion, we cannot actually predict stock prices, so we will try and predict trends


# We will predict 'open' prices


#Recurrent Neural Network


# Part-1 Data PreProcessing

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the Dataset
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scalling
# In case of an RNN or when there is sigmoid activation function,
# it is recommended to use Normalisation feature scaling
# as compared to Sta  ndardisation scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timestamps and 1 output
# 60 timesteps = at each time T, the RNN looks at stock prices under previous 60 operations
# This number can lead to overfitting or even underfitting if not chosen correctly
# X_train = input to the RNN, [contains prices for previous 60 days] y_train=output of the RNN 
X_train=[]
Y_train=[]

for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60: i, 0])
    Y_train.append(training_set_scaled[i, 0])
    
X_train, Y_train=np.array(X_train), np.array(Y_train)

# Reshaping 
# We can add new dimensions/indicators to make better predictions
# reshape function = to add a new dimension
# we also need to reshape to make them compatible as the input required for the RNN 
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



# Part-2 Building the RNN

# Importing the Libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#Initialising the RNN
regressor = Sequential()

# Add the first LSTM layer and some dropout regularisation
# units=number of memory cells in the LSTM, large number required as predicting
# stock price is a complex task
# return_sequences=True as we are making a stacked LSTM network
# input shape = last 2 from the X_train after reshaping
regressor.add(LSTM (units=50, return_sequences=True, input_shape=(X_train.shape[1], 1) ))
# 20% neurons of the LSTM network will be ignored during each iteration of training
regressor.add(Dropout(rate = 0.2))


# Add the second LSTM layer and some dropout regularisation
regressor.add(LSTM (units=50, return_sequences=True))
regressor.add(Dropout(rate = 0.2))

# Add the third LSTM layer and some dropout regularisation
regressor.add(LSTM (units=50, return_sequences=True))
regressor.add(Dropout(rate = 0.2))

# Add the fourth LSTM layer and some dropout regularisation
# False, as we are not gonna return anymore values
regressor.add(LSTM (units=50, return_sequences=False))
regressor.add(Dropout(rate = 0.2))

# Add the output layer
# We need a fully connected layer, so using Dense
regressor.add(Dense(units=1))

# Compile the RNN
# For recurrent neural network, RMSprop optimiser is recommended
# However, adam optimiser is a safe choice, and gives decent results in this case
regressor.compile(optimizer= 'adam', loss='mean_squared_error')

# Fit the RNN to the Training Set
regressor.fit(X_train, Y_train, epochs = 100, batch_size = 32)


# Part-3 Making predictions and result visualisation

# Getting the real stock prices of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock prices of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test=[]

for i in range(60, 80):
    X_test.append(inputs[i-60: i, 0])
    
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()