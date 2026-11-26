import numpy as np

array = np.arange(10, 50, 2)

print("Created Array:", array)


print("Data Type:", array.dtype)
print("Number of Dimensions:", array.ndim)
print("Array Shape:", array.shape)

matrix = array.reshape(4, 5)

print("4x5 Matrix:")
print(matrix)

print("New Matrix Shape:", matrix.shape)

selected_part = matrix[:2, -3:]

print("Selected Part:")
print(selected_part)


column_averages = np.mean(matrix, axis=0)

standard_deviation = np.std(matrix)

print("Column Averages:", column_averages)
print("Matrix Standard Deviation:", standard_deviation)
print(np.shares_memory(array, selected_part))