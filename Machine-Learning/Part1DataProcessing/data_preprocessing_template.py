# Data PreProcessing

# Importing Libraries
# It contains mathematical tools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv('Data.csv')
# print(dataset)
# on left : all the lines
# on right : columns except last one, because independent
# hai wo
X = dataset.iloc[:, :-1].values
# independent variable y, last column
y = dataset.iloc[:, 3].values
# print(y)

# Taking care of missing data
# sciket learn, libraries for ML models
from sklearn.preprocessing import Imputer

# missing values are called NaN, so put it there
# strategy is going to be mean
# axis is for mean from row or column
imputer = Imputer(missing_values='NaN',
                  strategy='mean',
                  axis=0)

#: means choose all
# take indexes from 1-2,
# index of missing data,
#  upper bound is excluded, so take 3
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
# print(X)
# Missing data replaced

# FOR CATEGORICAL DATA :
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()
# Put the encoded value in the X
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# Problem is that, these numbers
# will be used in the equations and 2>1>0,
# means greater priority, so error
# HENCE WE CHOOSE DUMMY ENCODING

# categorical features means which column, to hot encode
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
# print(X)
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# SPLITTING INTO TRAINING SET AND TEST SET
from sklearn.cross_validation import train_test_split

#Independent variables and the dependent
#test_size=0.2 means 20%, test_size+train_size=1.0, so will be redundant
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=0)
#FEATURE SCALING
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train=sc_X.fit_transform(X_train)
#we don't need to fit test 
X_test=sc_X.transform(X_test)
#