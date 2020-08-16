#K Means Clustering Algorithm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Spending Score = no. of times they visit the mall , and their income
#and based on various other criterion, we give them spending scores(1-100)
#Closer to 100, means it spends more
# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values
#We don't know what we are looking for and hence don't know the number of clusters

#USING ELBOW METHOD TO FIND NUMBER OF CLUSTERS
from sklearn.cluster import KMeans
#We will calculate 10 WCSS values by iterating
wcss=[]
for i in range(1,11):
    #n_clusters is number of clusters, init specifies the way of picking the centroids
    #random nahi karenge to avoid Random Initialisation Trap, we use kmeans++
    #maximum kitne baar centroid nikalne ki try kare? We specified 300 here
    #n_init se pata chalega number of times kmeans starts for different values of centroids for each i
    #random_state represents the seed for random number generation
    #wcss=within cluster sum of squares, also called inertia
    kmeans= KMeans(n_clusters=i,random_state=0,init='k-means++', max_iter=300, n_init=10)
    #Fit that
    kmeans.fit(X)
    #Calculate the wcss and add to the wcss array
    wcss.append(kmeans.inertia_)

#Plot to find the elbow
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

#We find an elbow at NoOfClusters=5

#Applying kMeans to the mall dataset

kmeans=KMeans(n_clusters=5, init='k-means++',max_iter=300, n_init=10, random_state=0)
#For each client, kmeans will predict the cluster it belongs to
y_kmeans=kmeans.fit_predict(X)

#Visualising the Clusters
#Cluster numbers are from 0-4
#Our first cluster is at y_means=0
#We said that X se, we want the entries with y_kmeans=0, i.e cluster 1
#By this we gave the X cordinate of all the entries in cluster 0
#            plt.scatter(X[y_kmeans==0, 0])

#s=100, means size=100 for our datapoints, c is for color
plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1],s=100,c='red', label='Cluster 0')#Careful
plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1],s=100,c='blue', label='Cluster 1')#Standard
plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1],s=100,c='green', label='Cluster 2')#Target
plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==3, 1],s=100,c='cyan', label='Cluster 3')#Careless
plt.scatter(X[y_kmeans==4, 0], X[y_kmeans==4, 1],s=100,c='magenta',label='Cluster 4')#Sensible

#TO PLOT THE CENTROIDS
#cluster_centers se center of clusters ke coordinate aa jate hain, and isliye 0(i.e X) and 1(i.e Y) coordinate
#alag kardiya
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()