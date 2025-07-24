import numpy as np
from src.simplex import solve

"""
First example
"""
A = np.array([[2, -2, 4, 1],
    [1, 1, 0, 1]])

b = np.array([2,0])

c = np.array([1, -1, 0, 1])

v = np.array([0, 0, 0.5, 0])

base_vectors_indices = [0,2]

result1 = solve(A, base_vectors_indices, b, c, v)

print(result1)

"""
Second example
"""

A = np.array([[1, 1, 3, 1],
              [1, -1, 1, 2]])

b = np.array([3,1])

c = np.array([1, 2, 3, 4])

v = np.array([2, 1, 0, 0])

base_vectors_indices = [0,1]

result2 = solve(A, base_vectors_indices, b, c, v)

print(result2)
