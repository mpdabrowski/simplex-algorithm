import numpy as np

class SimplexTable:
    def __init__(
            self, 
            width, 
            height,
            base_vectors_indices,
            non_base_vectors_indices,
            BAs
        ):
        self.simplex_table = np.array([[0]*(width) for _ in range(height + 1)], dtype=np.float64)

        k = 0
        for i in range(width):
            if i in base_vectors_indices:
                self.simplex_table[i][i] = 1.0
            
            if i in non_base_vectors_indices:
                self.simplex_table[:, i] = BAs[:, k]
                k += 1

    def get_table(self):
        return self.simplex_table
    
    def add_column(self, last_column):
        self.simplex_table = np.hstack((self.simplex_table, np.atleast_2d(last_column).T))