import numpy as np
from typing import List
from numpy.typing import NDArray

class SimplexTable:
    def __init__(
            self, 
            simplex_table
        ):
       self.simplex_table = simplex_table
        
    def get_table(self) -> NDArray[np.float64]:
        if not isinstance(self.simplex_table, np.ndarray):
            raise Exception("You need to build simplex table first")
        
        return self.simplex_table
    
    def add_column(self, last_column: NDArray[np.float64]) -> None:
        if not isinstance(self.simplex_table, np.ndarray):
            raise Exception("You need to build simplex table first")
        
        self.simplex_table = np.hstack((self.simplex_table, np.atleast_2d(last_column).T))

    def is_solvable(self, non_base_vectors_indices) -> bool:
        if not isinstance(self.simplex_table, np.ndarray):
            raise Exception("You need to build simplex table first")
        
        for i in non_base_vectors_indices:
            column = self.simplex_table[:, i]
            delta = column[-1]
            if delta > 0:
                for j in column[:-1]:
                    if j > 0:
                        return True
