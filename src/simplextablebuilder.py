import numpy as np
from typing import List
from numpy.typing import NDArray
from simplextable import SimplexTable

class SimplexTableBuilder:

    def build(
        self, 
        width: int, 
        height: int,
        base_vectors_indices: List[int],
        non_base_vectors_indices: List[int],
        BAs: NDArray[np.float64]
    ) -> SimplexTable:
        simplex_table = np.array([[0]*(width) for _ in range(height + 1)], dtype=np.float64)
        r = 0
        k = 0
        for i in range(width):
            if i in base_vectors_indices:
                simplex_table[r][i] = 1.0
                r += 1
            
            if i in non_base_vectors_indices:
                simplex_table[:, i] = BAs[:, k]
                k += 1
        
        
        return SimplexTable(simplex_table)
