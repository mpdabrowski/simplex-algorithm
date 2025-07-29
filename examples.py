import numpy as np
from src.simplex import Simplex

"""
First example
"""
A = np.array([[2, -2, 4, 1],
    [1, 1, 0, 1]])

b = np.array([2,0])

c = np.array([1, -1, 0, 1])

v = np.array([0, 0, 0.5, 0])

base_vectors_indices = [0,2]

simplex = Simplex(A,base_vectors_indices,b,c,v)

result1 = simplex.solve()

print("Result 1 = ", result1)

"""
Second example
"""

A = np.array([[1, 1, 3, 1],
              [1, -1, 1, 2]])

b = np.array([3,1])

c = np.array([1, 2, 3, 4])

v = np.array([2, 1, 0, 0])

base_vectors_indices = [0,1]

simplex = Simplex(A,base_vectors_indices,b,c,v)

result2 = simplex.solve()

print("Result 2 = ", result2)

"""
Third example
"""

A = np.array([[1, 1, 3, 1],
              [1, -1, 1, 2]])

b = np.array([3, 1])

c = np.array([1, 2, 3, 4])

v = np.array([0, 5/3, 0, 4/3])

base_vectors_indices = [1,3]

simplex = Simplex(A,base_vectors_indices,b,c,v)

result3 = simplex.solve()

print("Result 3 = ", result3)
