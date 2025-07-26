import numpy as np
from typing import List
from numpy.typing import NDArray

class SimplexTable:
    def __init__(
            self, 
            width: int, 
            height: int,
            base_vectors_indices: List[int],
            non_base_vectors_indices: List[int],
            BAs: NDArray[np.float64]
        ):
        self.width = width
        self.height = height
        self.base_vectors_indices = base_vectors_indices
        self.non_base_vectors_indices = non_base_vectors_indices
        self.BAs = BAs
        self.simplex_table = self.build()

    def build(self) -> NDArray[np.float64]:
        simplex_table = np.array([[0]*(self.width) for _ in range(self.height + 1)], dtype=np.float64)
        r = 0
        k = 0
        for i in range(self.width):
            if i in self.base_vectors_indices:
                simplex_table[r][i] = 1.0
                r += 1
            
            if i in self.non_base_vectors_indices:
                simplex_table[:, i] = self.BAs[:, k]
                k += 1
        
        self.simplex_table = simplex_table

        return self.simplex_table
        

    def get_table(self) -> NDArray[np.float64]:
        if not isinstance(self.simplex_table, np.ndarray):
            raise Exception("You need to build simplex table first")
        
        return self.simplex_table
    
    def add_column(self, last_column: NDArray[np.float64]):
        if not isinstance(self.simplex_table, np.ndarray):
            raise Exception("You need to build simplex table first")
        
        self.simplex_table = np.hstack((self.simplex_table, np.atleast_2d(last_column).T))

    def is_solvable(self):
        if not isinstance(self.simplex_table, np.ndarray):
            raise Exception("You need to build simplex table first")
        
        for i in self.non_base_vectors_indices:
            column = self.simplex_table[:, i]
            delta = column[-1]
            if delta > 0:
                for j in column[:-1]:
                    if j > 0:
                        return True
