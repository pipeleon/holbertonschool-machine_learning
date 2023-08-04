#!/usr/bin/env python3
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

data = np.load("data.npy")
labels = np.load("labels.npy")

data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(pca_data)):
    if labels[i] == 0:
        color = 'b'
    elif labels[i] == 1:
        color = 'r'
    else:
        color = 'y'
    ax.scatter(pca_data[i][0], pca_data[i][1], pca_data[i][2], color=color)

ax.set_xlabel('U1')
ax.set_ylabel('U2')
ax.set_zlabel('U3')
plt.title('PCA of Iris Dataset')
plt.plasma()
plt.savefig('101-pca.png')