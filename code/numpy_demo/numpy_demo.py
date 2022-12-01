"""
    Numpy demo
"""
# 1
import numpy as np

# 2
array = np.zeros((5, 5))
print(array)

# 3
shape = array.shape
nb_lines = array.shape[0]
nb_columns = array.shape[1]
print(shape)
print(f"{nb_lines} lines")
print(f"{nb_columns} columns")

# 4
for line in range(nb_lines):
    for column in range(nb_columns):
        array[line, column] = np.random.uniform(-1, 1)
print(array)

# 5
array = np.random.uniform(-1, 1, (5, 5))
print(array)

# 6
transposed_array = np.transpose(array)
print(array)
print("\n transposed array")
print(transposed_array)

# 7
print(array >= 0)

# 8
positive_indexes = np.where(array >= 0)
print(positive_indexes)

# 9
second_line_sum = 0
for column in range(nb_columns):
    second_line_sum += array[1, column]
print(second_line_sum)

# 10
print(array[1, :].sum())

# 11
print(array.sum(axis=1))
print(array.sum(axis=1)[1])

# 12
print(array.sum(axis=1)[1] == array[1, :].sum())

# 13
print(array.sum())

# 14
print((array >= 0))
print((array >= 0).all())
