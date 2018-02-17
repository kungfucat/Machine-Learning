#LOGISTIC REGRESSION
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
#Predicting on the basis of 2 and 3 column only, so humne aise likha
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 50/400, random_state = 0)

#WILL USE FEATURE SCALING FOR BETTER RESULTS HERE
# Feature Scaling
#NO scalling for y, since wo 0 ya 1 hai
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Fit logistic regression to training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
#fitted this way so that our classifier is able to learn the corelations between
#X_train and y_train
classifier.fit(X_train,y_train)

#Predict the test results
#y_pred is a vector of predictions
y_pred=classifier.predict(X_test)

#Making the confusion matrix for evaluation of model
#Not a class, just a function
#Ground Truth = Real Values that appeared
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred=y_pred)

# array([[65,  3],
#        [ 8, 24]])
#ye cm aaya, actually mein 65+24 correct predictions thi, i.e 89, while 11 were wrong

#Visualise the training set

# Visualising the Training set results
#We take all the pixels with 0.01 resolution
#just like each pixel is a user of our dataset
#for each of these pixels we predicted whether it is 0 or 1
#So bade wale regions mil gaye
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
#prepare the grid of pixels, -1 because we don't our points to be squeezed
#maximum for range of pixels
#same for salary
#resolution 0.01
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
#apply classifier to all the pixel points
#contour to make the contour between the two parts

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
#plot all the data points
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
#to specify which point is which i.e 0 is not taken
plt.legend()
plt.show()

#ANALYSIS OF GRAPH :
"""The points are the actual points of the training set
Green points = purchased
Red points = not purchased, X pe age, Y pe salary
Classifier will classify the users into one of the categories
The big coloured parts are called prediction regions
Points are the truth and the regions are the predictions
The two regions are seperated by a straight line called the prediction boundary
The prediction boundary is a straight line is the essence of Logistic Regression, because Linear classifier hai
The few points which are wrongly placed is because of this linear nature
Non linear classifiers can do a terrific job at this though (from the graph we realised)
"""


#Test set
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()