import unittest
import numpy as np
from src.tableau import Tableau 

class TableauTest(unittest.TestCase):
    def setUp(self):
        self.simplex_tableau = np.array([
            [1, 2, 0, 1],
            [0, 1, 1, 2],
            [0, -1, 1, 2]
        ])

        self.simplex_tableau_unsolvable = np.array([
            [1, -2, 0, -1],
            [0, -1, 1, -2],
            [0, 3, 1, 2]
        ])

        self.simplex_tableau_with_additional_column = np.array([
            [1, 2, 0, 1, 12],
            [0, 1, 1, 2, 3],
            [0, -1, 1, 2, 2]
        ])

    def test_add_column(self):
        tableau = Tableau(self.simplex_tableau)
        column = np.array([12, 3, 2])
        tableau.add_column(column)
        self.assertTrue(np.array_equal(tableau.simplex_tableau, self.simplex_tableau_with_additional_column))

    def test_is_solvable_for_solvable_tableau(self):
        tableau = Tableau(self.simplex_tableau)
        self.assertTrue(tableau.is_solvable([1, 3]))

    def test_is_solvable_for_unsolvable_tableau(self):
        tableau = Tableau(self.simplex_tableau_unsolvable)
        self.assertFalse(tableau.is_solvable([1, 3]))

if __name__ == '__main__':
    unittest.main()
