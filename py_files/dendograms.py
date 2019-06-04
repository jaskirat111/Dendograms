# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 14:53:48 2018

@author: DELL
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#ading of data
dataset=pd.read_csv('C:\\Users\\DELL\\Desktop\\ml\\Dendograms\\csv_files\\Mall_Customers.csv')

X=dataset.iloc[:, [3,4]].values
#
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++')
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
#using the dendogram to find the optimal 
import scipy.cluster.hierarchy as sch
dendogram=sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Euclidian distances')
plt.show()
#fittig  hierarchical clustering to the dataset
from sklearn.cluster import AgglomerativeClustering as ass
hc=ass(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc=hc.fit_predict(X)
y_kmeans=y_hc

#visualising the clusters
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='red',label='Cluster 1')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='blue',label='Cluster 2')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c='green',label='Cluster 3')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,c='cyan',label='Cluster 4')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100,c='magenta',label='Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income(k$)')
plt.ylabel('spending Score(1-100)')
plt.legend()
plt.show()
