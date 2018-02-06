# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#RANDOM STATE ENSURES WE GET THE SAME RESULT
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =1/3, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Fitting Simple Linear Regression To Training Set
from sklearn.linear_model import LinearRegression

regressor=LinearRegression()
#fit regressor to the dataset
#target value = dependent values
regressor.fit(X_train,y_train)

#predicting the test set results
#we will create vector of predicted values
#X_test is matrix of features
y_pred=regressor.predict(X_test)

#visualising the training set results
plt.scatter(X_train,y_train, color='red')
#we want to comare the x_train and the predicted
#values of Y_train
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Exp.(Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test,y_test, color='red') 
#this plot is the linear regression line
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Exp.(Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
