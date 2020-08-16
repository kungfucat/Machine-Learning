# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
#if we just use index :,1  toh wo vector samjhega, but we want it array of features hain
#toh array samjhe, so humne index aise likha, although ek hi column hai fir bhi
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, 2].values

#Since dataset is so small, we don't need to split in test and training set,
#waise bhi bluff detector hi bana rahe hain

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

#NO FEATURE SCALING NEEDED
# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Fitting Linear Regression Model
from sklearn.linear_model import LinearRegression
linearRegression=LinearRegression()
linearRegression.fit(X,y)


#Fitting Polynomial Regression Model
#polyFeatures transforms x to x^2 or x^3 or anything we want
from sklearn.preprocessing import PolynomialFeatures
polynomialFeatures=PolynomialFeatures(degree=4)
#first fit polyFeatures to X and then transform
X_poly = polynomialFeatures.fit_transform(X)
#ye fit_tranform, apne aap ek full 1's ka column daaldeta hai, for that constant b0, x0 ke saath wala

linearRegression2=LinearRegression()
linearRegression2.fit(X_poly, y)

#Visualise li""'""near regression

plt.scatter(X,y,color='red')
plt.plot(X, linearRegression.predict(X), color='blue')
plt.title('Truth or Bluff(Linear Regression)')
plt.xlabel('PositionLevel')
plt.ylabel('Salary')
plt.show()

#min, max and step size, but this is a vector, we want an array
X_grid=np.arange(min(X), max(X), 0.1)
#reshape vector into a matrix with rows=X_grid ki number of rows, and 1 column
X_grid= X_grid.reshape((len(X_grid), 1))
#Poly Regression can give complex polynomial functions as result,
#i.e anything but a straight line
#Visualise Poly regression
plt.scatter(X,y,color='red')
#polynomial wale regressor se, X_poly ko predict karao and usko plot kardo
#we could use X_poly directly in this regressor but next time se yahan buss X ko replace karna hoga
# plt.plot(X, linearRegression2.predict(polynomialFeatures.fit_transform(X)), color='blue')
#X ki jagah, use X_grid for plots
plt.plot(X_grid, linearRegression2.predict(polynomialFeatures.fit_transform(X_grid)), color='blue')
plt.title('Truth or Bluff(Polynomial Regression)')
plt.xlabel('PositionLevel')
plt.ylabel('Salary')
plt.show()
#4 degree ke baad, 1 ke steps mein straight lines aane lagi, so to improve it
#humne 0.1 ke gaps pe plot karenge instead of 1, toh curve continous aayega issey


#Predict with linear regression
#directly value daal sakte hain predict karne ke liye
linearRegression.predict(6.5)


#predict with polynomial regression
linearRegression2.predict(polynomialFeatures.fit_transform(6.5))