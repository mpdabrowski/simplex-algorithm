import numpy as np
from src.simplextable import SimplexTable 

def get_dash_value(c, base_column_indices):
    return c[base_column_indices]

def get_column_base(A, base_column_indices):
    B = np.atleast_2d(A[:, base_column_indices[0]]).T
    for i in base_column_indices[1:]:
        column = A[:, i]
        B = np.hstack((B, np.atleast_2d(column).T))

    return B

def is_v_solution(deltas):
    return len([i for i in deltas if i > 0]) == 0

def is_solvable(simplex_table, ind_zero):
    for i in ind_zero:
        column = simplex_table[:, i]
        delta = column[-1]
        if delta > 0:
            for j in column[:-1]:
                if j > 0:
                    return True
    
    return False

def get_bigger_than_zero_I(simplex_table, ind_zero):
    print("ind_zero", ind_zero)
    for i in ind_zero:
        column = simplex_table[:, i]
        delta = column[-1]
        if delta > 0:
            k = i
            print((k, column[:-1]))
            return (k, column[:-1])
        
def get_new_A(A, k, js, non_base_vectors_indices):
    A[:, js] = A[:, k]
    A = np.delete(A, np.s_[non_base_vectors_indices], 1)
    
    return A


def get_new_v_vector(base_vectors_indices, c, w):
    v_new = np.zeros(len(c))
    j = 0
    for i in base_vectors_indices:
        v_new[i] = w[j]
        j += 1
    return v_new


def solve(A, base_vectors_indices, b, c, v):

    while True:
        c_dash = get_dash_value(c, base_vectors_indices)
        v_dash = get_dash_value(v, base_vectors_indices)
        B = get_column_base(A,base_vectors_indices)
        non_base_vectors_indices = [i for i in range(len(c)) if i not in base_vectors_indices]
        print(B)
        B = np.linalg.matrix_power(B, -1)
        deltas = []
        BAs = np.array([])
        for i in non_base_vectors_indices:
            BA = np.matmul(B, A[:, i])
            if BAs.size == 0:
                BAs = np.atleast_2d(BA).T
            else:
                BAs = np.hstack((BAs, np.atleast_2d(BA).T))
            dot_prod = np.dot(c_dash, BA)
            delta = dot_prod - c[i]
            deltas.append(delta)

        BAs = np.vstack([BAs, np.array(deltas)])
        simplex_table = SimplexTable(
            len(c), 
            len(c_dash),
            base_vectors_indices,
            non_base_vectors_indices,
            BAs
            ).get_table()

        print(simplex_table)
        v_dash = np.hstack((v_dash, np.dot(c, v)))
        last_column = np.array([v_dash])
        simplex_table = np.hstack((simplex_table, np.atleast_2d(last_column).T))
        print(simplex_table)

        if is_v_solution(deltas):
            return v
        
        if False == is_solvable(simplex_table, non_base_vectors_indices):
            raise Exception("This problem does not have solution")
        
        k, I = get_bigger_than_zero_I(simplex_table, non_base_vectors_indices) # last element of new base

        v_dash = get_dash_value(v, base_vectors_indices)
        minim = float('inf')
        min_inx = 0
        for i, val in enumerate(I):
            if val <= 0:
                continue
            val = v_dash[i] / val
            if val < minim:
                minim = val
                min_inx = i

        js = min_inx 

        A_new = get_new_A(A,k,js, non_base_vectors_indices)
        base_vectors_indices[js] = k
        w = np.linalg.solve(A_new, b)

        v_new = get_new_v_vector(base_vectors_indices, c, w)

        v = np.array(v_new)
        

    raise Exception('Error while solving problem')
