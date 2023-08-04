#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

people = ['Farrah', 'Fred', 'Felicia']

fruit_counts = {
    'apples': fruit[0],
    'bananas': fruit[1],
    'oranges': fruit[2],
    'peaches': fruit[3]
}

width = 0.5

fig, ax = plt.subplots()
bottom = np.zeros(3)

for fruit, fruit_count in fruit_counts.items():
    p = ax.bar(people, fruit_count, width, label=fruit, bottom=bottom)
    bottom += fruit_count

    ax.bar_label(p, label_type='center')

ax.set_title('Number of Fruit per Person')
ax.legend()

plt.savefig('6-bars.png')
