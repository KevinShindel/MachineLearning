# Clustering 
- Divide customer base into clusters such that:
- - homogenous within cluster is maximized (cohesive)
- - heterogeneous between clusters is maximized (separated)
- Example business analytics applications:
- - targeted marketing or advertising
- - allocating marketing resources/ mass customization
- - differentiating between brands
- - identifying most profitable customers
- - clustering claims in an insurance setting
- - clustering cash transfers for anti-money laundering

# Hierarchical Clustering
x1 = 50, x2 = 30, y1 = 20, y2 = 10
- Euclidean distance = sqrt((x1-x2)^2 + (y1-y2)^2) = sqrt((50-30)^2 + (20-10)^2) = sqrt(400 + 100) = sqrt(500) = 22.36
- Manhattan distance = |x1-x2| + |y1-y2| = |50-30| + |20-10| = 20 + 10 = 30

Type of linkage:
- Single linkage: minimum distance between clusters
- Complete linkage: maximum distance between clusters
- Average linkage: average distance between clusters
- Centroid linkage: distance between centroids of clusters

# Hierarchical clustering in Python
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering

# Create a simple 2d data set
X, y = make_blobs(centers=[[0, 0], [1, 1], [-1, 1]], cluster_std=[0.3, 0.3, 0.1])
plt.scatter(X[:,0], X[:,1])

# Hierarchical clustering

clf = AgglomerativeClustering(n_clusters=3)
clusters = clf.fit_predict(X)
plt.scatter(X[:,0], X[:,1], c=clusters+1)
plt.show()
```

# K-means Clustering
- Select k-observations as initial cluster centroids
- Assign each observation to the nearest centroid
- When all observations are assigned, recalculate the centroids
- Repeat until cluster centroids no longer change

# k-means clustering in Python
```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Create a simple 2d data set
X, y = make_blobs(centers=[[0, 0], [1, 1], [-1, 1]], cluster_std=[0.3, 0.3, 0.1])
plt.scatter(X[:,0], X[:,1])

# K-means

clf = KMeans(n_clusters=3)
clusters = clf.fit_predict(X)
plt.scatter(X[:,0], X[:,1], c=clusters+1)
plt.show()
```

# DBSCAN
- Density-based spatial clustering of applications with noise
- Two points p and q are density-connected if they are commonly density-reachable from a point o
- Density-connectivity is symmetric but not transitive
- Basic intuition: cluster observations in high-density regions, mark observations in low-density regions as noise (outliers)
- MinPts: minimum number of points in a neighborhood for a point to be considered a core point
- Types of points:
- - Core point: have at least MinPts points in their neighborhood
- - Border point: have fewer than MinPts points in their neighborhood but are reachable from a core point
- - Outlier point: neither core nor border points
- Simplified algorithm:
- - Randomly select a point p from the dataset
- - Retrieve all points density-reachable from p w.r.t. Eps and MinPts
- - If p is a core point, a cluster is formed
- - If p is a border point, no points are density-reachable from p, mark p as noise
- - Repeat until all points have been processed

# DBSCAN in Python
```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN

# Create a simple 2d data set
X, y = make_blobs(centers=[[0, 0], [1, 1], [-1, 1]], cluster_std=[0.3, 0.3, 0.1])
plt.scatter(X[:,0], X[:,1])

# DBSCAN

clf = DBSCAN(eps=0.3)
clusters = clf.fit_predict(X)
plt.scatter(X[:,0], X[:,1], c=clusters+1)
plt.show()
```

# Evaluating Clustering Solutions

- Evaluating a cluster solution is not trivial
- Sum of squared errors (SSE) is not a good measure

# Dendrogram
- Used to visualize hierarchical clustering
- Each leaf represents an observation
- Height of the leaf represents the distance between observations
- To decide the number of clusters, cut the dendrogram at a certain height