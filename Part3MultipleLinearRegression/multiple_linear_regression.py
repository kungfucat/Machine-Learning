# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#Encoding the categorical data
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
labelEncoder_X=LabelEncoder()
X[:,3]=labelEncoder_X.fit_transform(X[:,3])
#one hot encoder needs label encoded first
#So used label Encoder first, and then oneHotEncoder
onehotEncoder=OneHotEncoder(categorical_features=[3])
X=onehotEncoder.fit_transform(X).toarray()

#Avoiding the dummy variable trap
#Just removed the column zero
#multiple regression library takes care of it, but for some libraries
#we will need to physically remove a column to avoid the trap
X=X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#NO NEED OF FEATURE SCALING BECAUSE THE LIBRARY TAKES CARE OF THAT FOR US

#Fitting Multiple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#Predicting the test set results
y_pred=regressor.predict(X_test)

#Building the optimal model using backward elimination
import statsmodels.formula.api as sm

#y=b0x0+b1x1+b2x2+...+bnxn, b0 is constant, and x0=1
#statsmodels library doesn't take care of this, so we will have to
#linear regresssion library knows that this b0 constant is there
#so we insert a row of 1's (x0=1 ho taaki)
#np.ones returns a matrix of 1's
#shape is the shape of the matrix to form, 50 rows, 1 column
# X = np.append(arr=X,values=np.ones(shape=(50, 1)))

# .astype is to prevent a datatype error
#axis=0 for a line, axis=1 for column

#we are appending the values array to arr, but if we want
#that constants to appear in the first column, we add X to
#an array of 50 1s
# X = np.append(values=np.ones((50, 1)).astype(int),arr=X,axis=1)
X = np.append(arr=np.ones((50, 1)).astype(int),values=X,axis=1)
#the optimal X
#we are gonna remove each column one by one, so we used
#all rows and each column number here
X_opt = X[: , [0,1,2,3,4,5]]

#we will use regressor from the ols library and not the LinearRegression one
#ols : ordinary least squares
#exog clearly mentions that intercept(column of 1's) is not included, so need to
#do it manually,
#endog is the dependent variable
#.fit will fit the ols to x_opt and y
#STEP 2 : fit the full model with all possible predictors
#a functions called summary of statsModelsLibrary, returns the
#summary
#LOWER THE P VALUE, more significant it is, w.r.t  dependent variable
regressor_OLS=sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

#Now we will remove the variable with very high p values, e.g index 2
#Column to remove ke liye, look in original X, because we are using that
#only here
X_opt=X[:,[0,1,3,4,5]]
regressor_OLS=sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

#P value cannot be 0, 0.000 shows a very very small value
X_opt=X[:,[0,3,4,5]]
regressor_OLS=sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()


X_opt=X[:,[0,3,5]]
regressor_OLS=sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

#R Squared and Adjusted R Squared will help us decide
#whether to keep a variable in the model or not

X_opt=X[:,[0,3]]
regressor_OLS=sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()