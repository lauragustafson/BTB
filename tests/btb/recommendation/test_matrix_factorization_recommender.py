from unittest import TestCase

import numpy as np
from mock import patch

from btb.recommendation.matrix_factorization import MFRecommender


class TestBaseRecommender(TestCase):
    def setUp(self):
        self.n_components = 3
        self.dpp_decomposed = np.array([
            [.0, .2, .4],
            [.8, .6, .4],
            [1.0, .8, 0.9],
            [.6, 1.0, 0.8],
        ])
        self.dpp_ranked = np.array([
            [1, 2, 3],
            [3, 2, 1],
            [3, 1, 2],
            [1, 3, 2],
        ])
        # Set-up
        self.dpp_matrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 3, 0, 4, 6, 2, 5, 0, 8, 0],
            [0, 4, 0, 6, 0, 4, 0, 2, 1, 0, 0, 2, 3, 1, 0, 0],
            [1, 0, 1, 2, 0, 0, 6, 1, 0, 5, 1, 0, 0, 0, 0, 1],
            [0, 2, 3, 0, 0, 0, 0, 0, 4, 1, 3, 2, 0, 0, 1, 4]
        ])

    @patch('btb.recommendation.matrix_factorization.NMF')
    def test___init__(self, nmf_mock):
        nmf_mock().fit_transform.return_value = self.dpp_ranked
        # Run
        recommender = MFRecommender(self.dpp_matrix, self.n_components)
        np.testing.assert_array_equal(recommender.dpp_matrix, self.dpp_matrix)
        np.testing.assert_array_equal(recommender.dpp_ranked, self.dpp_ranked)
        assert recommender.n_components == self.n_components
        assert recommender.matching_dataset is None
        # assert dpp_vector has same number of entries as pipelines
        assert recommender.dpp_vector.shape[0] == self.dpp_matrix.shape[1]

    @patch('btb.recommendation.matrix_factorization.NMF')
    def test_fit(self, nmf_mock):
        nmf_mock().fit_transform.return_value = self.dpp_ranked
        n_components = 3
        X = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # Run
        recommender = MFRecommender(self.dpp_matrix, n_components)
        for i in range(self.dpp_ranked.shape[0]):
            nmf_mock().transform.return_value = self.dpp_ranked[i, :]
            recommender.fit(X)
            np.testing.assert_array_equal(
                recommender.matching_dataset,
                self.dpp_matrix[i],
            )

    def test_predict_one(self):
        # Run
        recommender = MFRecommender(self.dpp_matrix, self.n_components)
        # matching row is [1, 0, 0, 0, 0, 0, 0, 3, 0, 4, 6, 2, 5, 0, 8, 0]
        recommender.matching_dataset = self.dpp_matrix[0]

        indicies = [0]
        predictions = recommender.predict(indicies)
        expected = [1]
        np.testing.assert_array_equal(predictions, expected)

    def test_predict_all_matching(self):
        # Run
        recommender = MFRecommender(self.dpp_matrix, self.n_components)
        # matching row is [1, 0, 0, 0, 0, 0, 0, 3, 0, 4, 6, 2, 5, 0, 8, 0]
        recommender.matching_dataset = self.dpp_matrix[0]
        indicies = [1, 2, 3, 4, 5]
        predictions = recommender.predict(indicies)
        expected = [1, 1, 1, 1, 1]
        np.testing.assert_array_equal(predictions, expected)

    def test_predict_multiple_rankings(self):
        # Run
        recommender = MFRecommender(self.dpp_matrix, self.n_components)
        # matching row is [1, 0, 0, 0, 0, 0, 0, 3, 0, 4, 6, 2, 5, 0, 8, 0]
        recommender.matching_dataset = self.dpp_matrix[0]
        indicies = [0, 1, 14, 13]
        predictions = recommender.predict(indicies)
        expected = [2, 1, 3, 1]
        np.testing.assert_array_equal(predictions, expected)
