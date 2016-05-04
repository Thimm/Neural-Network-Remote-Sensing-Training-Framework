import numpy as np
from random import shuffle
percentage = 0.1
a = list(tuple(open('HDF5_Test/dataset-list.txt', 'r')))

shuffle(a)
per_items = int(len(a)*percentage)
print len(a[:per_items]), len(a[per_items:])
a_test = a[per_items:]
a_train = a[:per_items]
with open('shuffled_train.txt', 'w') as b:
	for item in a_test:
		b.write(item)

with open('shuffled_test.txt', 'w') as b:
	for item in a_train:
		b.write(item)
