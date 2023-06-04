import unittest
from main import Optimizer, Route, Customer, is_feasible, calculate_distance
import os
import random
from unittest.mock import MagicMock, patch
import numpy as np

class TestOptimizerMethods(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.optimizer = Optimizer(level_max=8)
        cls.customers = [Customer(i, (np.random.rand(), np.random.rand()), 0, 0, 0) for i in range(10)]
        cls.route1 = Route([
            Customer(1, (16, 23), 0, 408, 0),
            Customer(2, (22, 4), 62, 68, 0),
            Customer(3, (12, 6), 181, 205, 0)
        ])
        cls.route1.path = [1, 2, 3]
        cls.route2 = Route([
            Customer(1, (16, 23), 0, 408, 0),
            Customer(2, (22, 4), 62, 68, 0),
            Customer(3, (12, 6), 181, 205, 0),
            Customer(4, (47, 38), 306, 324, 0),
            Customer(5, (11, 29), 214, 217, 0)
        ])
        cls.route2.path = [1, 2, 3, 4, 5]

    # def test_perform_perturbation(self):
    #     path = list(range(10))  # initial path is [0, 1, 2, ..., 9]
    #     level = 2
    #     new_route = self.optimizer.perform_perturbation(level, path, self.customers)

    #     # Test that the length of the path is still 10
    #     self.assertEqual(len(new_route.path), 10)

    #     # Test that the path contains the same elements (regardless of order)
    #     self.assertCountEqual(new_route.path, path) 
        
    def test_is_feasible(self):
        self.assertTrue(is_feasible(self.route1))
        self.assertFalse(is_feasible(self.route2))

    # def test_calculate_cost(self):
    #     self.assertEqual(round(self.optimizer.calculate_cost([1,2,3], self.route1.customers)), 181)
    #     self.assertEqual(round(self.optimizer.calculate_cost([1,2,3,4,5], self.route2.customers)), round(343.1079506))

    # def test_calculate_distance(self):
    #     point1 = (1, 1)
    #     point2 = (4, 5)
    #     expected_distance = 5  # calculated manually using the Euclidean distance formula
    #     actual_distance = calculate_distance(point1, point2)
    #     self.assertEqual(actual_distance, expected_distance)
        
    # def test_random_solution(self):
    #     # Ensure seed is set for reproducibility
    #     random.seed(1)
    #     shuffle_ids = list(range(1,6))

    #     # Get the output from the method
    #     solution = self.optimizer.random_solution(shuffle_ids)
        
    #     # Check the output is a list
    #     self.assertTrue(isinstance(solution, list), "Output should be a list")

    #     # Check the output list has the same length as the input
    #     self.assertEqual(len(solution), len(shuffle_ids), "Output list should have the same length as the input list")

    #     # Check that all elements in the input list are in the output list (i.e., no elements were lost or added)
    #     self.assertTrue(all(elem in solution for elem in shuffle_ids), "All input elements should be in the output list")

    #     # Check that the output list is not the same as the input list (i.e., some shuffling has occurred)
    #     # Note: this isn't a perfect check, because it's possible (though unlikely) for shuffling to return the original order
    #     self.assertNotEqual(solution, shuffle_ids, "Output list should be shuffled (not in the original order)")

   
        
if __name__ == '__main__':
    unittest.main()
