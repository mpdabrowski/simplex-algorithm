import numpy as np
from .tableau_factory import TableauFactory
from .base_factory import BaseFactory

class Simplex:
    def __init__(
        self,
        A,
        base_vectors_indices,
        b,
        c, 
        v        
    ):
        self.A = A
        self.base_vectors_indices = base_vectors_indices
        self.b = b
        self.c = c
        self.v = v
        self.set_non_base_vector_indices()
        
    def solve(self, show_tableau = False):

        while True:
            c_dash = self.c[self.base_vectors_indices]
            v_dash = self.v[self.base_vectors_indices]
            base = BaseFactory(self.non_base_vectors_indices, self.A).build()

            simplex_tableau = TableauFactory().build(
                self.base_vectors_indices,
                self.non_base_vectors_indices,
                base.get_base_multiplied_by_A(c_dash, self.c)
            ).add_column(np.array([np.hstack((v_dash, np.dot(self.c, self.v)))]))
            
            if show_tableau:
                print(simplex_tableau)

            if simplex_tableau.is_solution():
                return self.v
            
            if False == simplex_tableau.is_solvable(self.non_base_vectors_indices):
                raise Exception("Cannot solve this problem")
            
            in_index, out_index = self.get_in_and_out_column(v_dash, simplex_tableau) 
            self.set_new_base_vector_indices(in_index, out_index)
            self.set_non_base_vector_indices()
            self.set_new_v()
            

        raise Exception('Error while solving problem')

    def set_new_v(self):
        v_new = self.get_new_v_vector(np.linalg.solve(self.A[:, self.base_vectors_indices.tolist()], self.b))
        self.v = np.array(v_new)

    def get_in_and_out_column(self, v_dash, simplex_tableau):
        k, I = simplex_tableau.get_bigger_than_zero_I(self.non_base_vectors_indices)

        minim = float('inf')
        min_inx = 0
        for i, val in enumerate(I):
            if val <= 0:
                continue
            val = v_dash[i] / val
            if val < minim:
                minim = val
                min_inx = i

        js = self.base_vectors_indices[min_inx]
        return k,js

    def set_non_base_vector_indices(self):
        self.non_base_vectors_indices = [i for i, _ in enumerate(self.c) if i not in self.base_vectors_indices]
 
    def set_new_base_vector_indices(self, k, js):
        base_vectors_indices = np.array(self.base_vectors_indices)
        out = np.where(base_vectors_indices == js)
        base_vectors_indices[out[0]] = k

        self.base_vectors_indices = np.sort(base_vectors_indices)

    def get_new_v_vector(self, w):
        v_new = np.zeros(len(self.c))
        j = 0
        for i in self.base_vectors_indices:
            v_new[i] = w[j]
            j += 1
        return v_new