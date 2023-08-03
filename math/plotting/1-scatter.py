#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x, y = np.random.multivariate_normal(mean, cov, 2000).T
y += 180

plt.scatter(x, y, c='#c623cb')
plt.suptitle("Men's Height vs Weight")
plt.xlabel('Height (in)')
plt.ylabel('Weight (lbs)')
plt.savefig("1-scatter.png")