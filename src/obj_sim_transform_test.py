import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import kmeans_plusplus
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import math

X = []
# mean
theta = np.pi
alpha = np.pi/10

rot_alpha = np.array([[math.cos(alpha), -math.sin(alpha), 0],
                     [math.sin(alpha), math.cos(alpha), 0],
                     [0, 0, 1]])
for i in range(10):
    for j in range(10):
        for k in range(10):
            pos = np.array([i, j, k])


            X.append(rot_alpha @ pos)

# for i in range(10):
#     for j in range(10):
#         X.append(np.array([i, j, 100]))

X = np.array(X)
# centers_init, indices = kmeans_plusplus(X, n_clusters=2, random_state=0)
# print(indices)
kmeans = KMeans(n_clusters=10, random_state=0, n_init="auto").fit(X)
# print(kmeans.labels_)

ax = plt.figure().add_subplot(projection='3d')
ax.scatter(X[:,0], X[:,1], X[:,2], c=kmeans.labels_)
plt.show()