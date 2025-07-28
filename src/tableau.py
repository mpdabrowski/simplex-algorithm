import numpy as np
from typing import List
from numpy.typing import NDArray

class Tableau:
    def __init__(
            self, 
            simplex_tableau
        ):
       self.simplex_tableau = simplex_tableau
    
    def add_column(self, last_column: NDArray[np.float64]) -> None:
        if not isinstance(self.simplex_tableau, np.ndarray):
            raise Exception("You need to build simplex table first")
        
        self.simplex_tableau = np.hstack((self.simplex_tableau, np.atleast_2d(last_column).T))

    def is_solvable(self, non_base_vectors_indices) -> bool:
        if not isinstance(self.simplex_tableau, np.ndarray):
            raise Exception("You need to build simplex table first")
        
        for i in non_base_vectors_indices:
            column = self.simplex_tableau[:, i]
            delta = column[-1]
            if delta > 0:
                for j in column[:-1]:
                    if j > 0:
                        return True
                    
    def get_bigger_than_zero_I(self, ind_zero):
        if not isinstance(self.simplex_tableau, np.ndarray):
            raise Exception("You need to build simplex table first")
        
        for i in ind_zero:
            column = self.simplex_tableau[:, i]
            delta = column[-1]
            if delta > 0:
                k = i
                return (k, column[:-1])
            
    def __str__(self):
        return f"Simplex tableau: \n {self.simplex_tableau}"
