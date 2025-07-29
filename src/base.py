import numpy as np
from typing import List
from numpy.typing import NDArray

class Base:
    def __init__(
            self,
            base: NDArray,
            non_base_vectors_indices: List, 
            A: NDArray, 
        ):
        self.base = base
        self.non_base_vectors_indices = non_base_vectors_indices
        self.A = A

    def get_base_multiplied_by_A(
            self, 
            c_dash: List, 
            c: List
        ) -> NDArray:
        inverted_base = self.get_inverted_base()
        deltas = []
        BAs = np.array([])
        for i in self.non_base_vectors_indices:
            BA = np.matmul(inverted_base, self.A[:, i])
            if BAs.size == 0:
                BAs = np.atleast_2d(BA).T
            else:
                BAs = np.hstack((BAs, np.atleast_2d(BA).T))
            dot_prod = np.dot(c_dash, BA)
            delta = dot_prod - c[i]
            deltas.append(delta)

        BAs = np.vstack([BAs, np.array(deltas)])
        return BAs
    
    def get_inverted_base(self) -> NDArray:
        return np.linalg.matrix_power(self.base, -1)
