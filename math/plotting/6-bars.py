#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

people = ['Farrah', 'Fred', 'Felicia']

fruit_counts = {
    'apples': [fruit[0], 'r'],
    'bananas': [fruit[1], 'y'],
    'oranges': [fruit[2], '#ff8000'],
    'peaches': [fruit[3], '#ffe5b4']
}

width = 0.5

fig, ax = plt.subplots()
bottom = np.zeros(3)

for fruit, fruit_count in fruit_counts.items():
    p = ax.bar(people, fruit_count[0], width, label=fruit, bottom=bottom, color=fruit_count[1])
    bottom += fruit_count[0]

ax.set_title('Number of Fruit per Person')
ax.legend()

plt.ylim([0, 80])
plt.ylabel('Quantity of Fruit')
plt.savefig('6-bars.png')
