import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Mall_Customers.csv')

X = dataset.iloc[:, [3, 4]].values

# Finding optimal number of clusters using dendogram
import scipy.cluster.hierarchy as sch

# linkage is actually the algorithm of the HC
# passed X to it
# method = ward , its a method reduce variance within each cluster
# similar to how we wanted to reduce wcss(within cluster sum of squares) earlier
dendogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('dendogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()

# FOR FINDING OPTIMAL NUMBER OF CLUSTERS
# Look at the longest vertical line, we can find in the graph
# without crossing any horizontal line
# this longest vertical line has 2 horizontal lines at its ends
# Count number of vertical lines between y and this line
# The number of vertical lines between these 2 horizontal lines is our answeer


# Fitting hierarchical clustering to the dataset
# Agglomerative use karenge

from sklearn.cluster import AgglomerativeClustering

# affinity is distance to do the linkage
clustering = AgglomerativeClustering(n_clusters=5,
                                     affinity='euclidean',
                                     linkage='ward')
y_hc = clustering.fit_predict(X)

#y_hc contains the cluster number for each customer

#Visualise the clusters

# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()