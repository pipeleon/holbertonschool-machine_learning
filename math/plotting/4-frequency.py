#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

plt.hist(student_grades, bins=range(0, 110, 10), edgecolor='black')
plt.xlim([0, 100])
plt.suptitle("Project A")
plt.xlabel('Grades')
plt.ylabel('Number of Students')
plt.savefig("4-frequency.png")