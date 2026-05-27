import unittest
import main
import numpy as np

class Tests(unittest.TestCase):

    def test_sort_points(self):
        r = np.array([
                [3, 4],
                [0, 4],
                [0, 0],
                [3, 0],
                ])

        r_sorted = main.sort_points(r)
        np.testing.assert_array_equal(r_sorted, np.array([
                [0, 0],
                [0, 4],
                [3, 4],
                [3, 0],
            ]))

if __name__ == "__main__":
    unittest.main()
