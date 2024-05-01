import sys; import os; sys.path.insert(1, os.path.join(os.getcwd(), "src"))


import unittest

import numpy as np

from distances_lookup import distance


class TestDistance(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_distance_shape_1(self):
        N_KEYS = 30
        WIDTH = MAX_COORD = 1080
        HEIGHT = 667
        dots = np.indices((WIDTH, HEIGHT)).transpose(1, 2, 0)  # (WIDTH, HEIGHT, 2)
        centers = np.random.randint(0, MAX_COORD, (N_KEYS, 2))
        result = distance(dots, centers)
        self.assertEqual(result.shape, (WIDTH, HEIGHT, N_KEYS))

    def test_distance_shape_2(self):
        N_KEYS = 30
        MAX_COORD = 1080
        DOT_DIMS = (100, 50, 90)
        dots = np.random.randint(0, MAX_COORD, (*DOT_DIMS, 2))
        centers = np.random.randint(0, MAX_COORD, (N_KEYS, 2))
        result = distance(dots, centers)
        self.assertEqual(result.shape, (*DOT_DIMS, N_KEYS))

    def testcase_dots_1(self):
        dots = np.array([[1, 2], [3, 4], [5, 6]])
        centers = np.array([[1, 2], [3, 4]])
        result = distance(dots, centers)
        expected = np.array([[0, 8], [8, 0], [32, 8]])
        self.assertTrue(np.allclose(result, expected))

    def testcase_grid_1(self):
        WIDTH = 2
        HEIGHT = 3
        dots = np.indices((WIDTH, HEIGHT)).transpose(1, 2, 0)
        centers = np.array([[4, 6]])
        result = distance(dots, centers)
        expected = np.array([[[52], [41], [32]], [[45], [34], [25]]])
        self.assertTrue(np.allclose(result, expected))


if __name__ == '__main__':
    unittest.main()
