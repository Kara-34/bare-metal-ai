
import numpy as np

# 1. Matrix Multiplication (@) vs. Element-wise Multiplication (*)

A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

B = np.array([
    [7, 8],
    [9, 10],
    [11, 12]
])

print("A @ B result:")
print(A @ B)

# A * B raises a ValueError because the shapes
# (2, 3) and (3, 2) are incompatible for broadcasting.

try:
    print(A * B)
except ValueError as e:
    print("ValueError:", e)


# 2. Transpose (.T)

A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

C = np.array([
    [7, 8, 9],
    [10, 11, 12]
])

print("\nA @ C.T result:")
print(A @ C.T)


# 3. Broadcasting (Without Loops)

matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

row = np.array([10, 20, 30])

print("\nAdding a row array:")
print(matrix + row)

column = np.array([
    [10],
    [20],
    [30]
])

print("\nAdding a column vector:")
print(matrix + column)


# 4. Boolean Masking

random_matrix = np.random.randint(1, 101, size=(3, 4))

print("\nOriginal random matrix:")
print(random_matrix)

# Replace all elements greater than 50 with 0 in one line.
random_matrix[random_matrix > 50] = 0

print("\nMatrix after Boolean Masking:")
print(random_matrix)
