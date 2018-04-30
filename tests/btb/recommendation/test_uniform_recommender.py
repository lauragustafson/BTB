import itertools
from unittest import TestCase

import numpy as np
from mock import patch

from btb.recommendation.uniform import UniformRecommender


class TestBaseRecommender(TestCase):
    def setUp(self):
        # Set-up
        self.dpp_matrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 3, 0, 4, 6, 2, 5, 0, 8, 0],
            [0, 4, 0, 6, 0, 4, 0, 2, 1, 0, 0, 2, 3, 1, 0, 0],
            [1, 0, 1, 2, 0, 0, 6, 1, 0, 5, 1, 0, 0, 0, 0, 1],
            [0, 2, 3, 0, 0, 0, 0, 0, 4, 1, 3, 2, 0, 0, 1, 4]
        ])
        self.dpp_matrix_small = np.array([
            [1, 0, 0, 0],
            [0, 4, 0, 6],
            [1, 0, 1, 2],
            [0, 2, 3, 0]
        ])

    def test_predict_all(self):
        indicies = np.array(range(4))
        permutation = [3, 2, 1, 4]
        recommender = UniformRecommender(self.dpp_matrix_small)
        with patch(
            'numpy.random.permutation',
            return_value=permutation
        ) as mock_random:
            predictions = recommender.predict(indicies)
        np.testing.assert_array_equal(
            permutation,
            predictions,
        )

    def test_predict_one(self):
        indicies = np.array([0])
        permutation = [1]
        recommender = UniformRecommender(self.dpp_matrix)
        with patch(
            'numpy.random.permutation',
            return_value=permutation
        ) as mock_random:
            predictions = recommender.predict(indicies)
        np.testing.assert_array_equal(
            permutation,
            predictions,
        )
