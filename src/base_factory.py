import numpy as np
from typing import List
from numpy.typing import NDArray
from .base import Base

class BaseFactory:
    def __init__(
            self,
            base_vectors_indices: List,
            A: NDArray
        ):
        self.base_vectors_indices = base_vectors_indices
        self.A = A

    def build(self):
        base = np.atleast_2d(self.A[:, self.base_vectors_indices[0]]).T
        for i in self.base_vectors_indices[1:]:
            column = self.A[:, i]
            base = np.hstack((base, np.atleast_2d(column).T))

        return Base(base, self.base_vectors_indices, self.A)
