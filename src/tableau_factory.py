import numpy as np
from typing import List
from numpy.typing import NDArray
from .tableau import Tableau

class TableauFactory:

    def build(
        self, 
        base_vectors_indices: List[int],
        non_base_vectors_indices: List[int],
        BAs: NDArray[np.float64]
    ) -> Tableau:
        num_columns = len(base_vectors_indices) + len(non_base_vectors_indices)
        num_rows = len(non_base_vectors_indices)
        simplex_tableau = np.array([[0]*(num_columns) for _ in range(num_rows + 1)], dtype=np.float64)
        r = 0
        k = 0
        for i in range(num_columns):
            if i in base_vectors_indices:
                simplex_tableau[r][i] = 1.0
                r += 1
            
            if i in non_base_vectors_indices:
                simplex_tableau[:, i] = BAs[:, k]
                k += 1
        
        
        return Tableau(simplex_tableau)
