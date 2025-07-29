import numpy as np
from typing import List, Tuple, Self
from numpy.typing import NDArray

class Tableau:
    def __init__(
            self, 
            simplex_tableau: NDArray[np.float64]
        ):
       self.simplex_tableau = simplex_tableau
    
    def add_column(self, last_column: NDArray[np.float64]) -> Self:
        self.simplex_tableau = np.hstack((self.simplex_tableau, np.atleast_2d(last_column).T))

        return self

    def is_solvable(self, non_base_vectors_indices) -> bool:
        for i in non_base_vectors_indices:
            column = self.simplex_tableau[:, i]
            delta = column[-1]
            if delta > 0:
                for j in column[:-1]:
                    if j > 0:
                        return True
                    
        return False
                    
    def is_solution(self) -> bool:
        last_row = self.simplex_tableau[-1]
        
        return len([i for i in last_row[:-1] if i > 0]) == 0
                    
    def get_bigger_than_zero_I(self, ind_zero: List[int]) -> Tuple[int, List]:
        for i in ind_zero:
            column = self.simplex_tableau[:, i]
            delta = column[-1]
            if delta > 0:
                k = i
                return (k, column[:-1])
        
        return None
            
    def __str__(self) -> str:
        return f"Simplex tableau: \n {self.simplex_tableau}"
