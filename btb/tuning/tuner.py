import logging
from builtins import object, range

import numpy as np

logger = logging.getLogger('btb')


class BaseTuner(object):
    def __init__(self, tunables, gridding=0, **kwargs):
        """
        Args:
            tunables: Ordered list of hyperparameter names and metadata
            objects. These describe the hyperparameters that this Tuner will
                be tuning. e.g.:
                [('degree', HyperParameter(type='INT', range=(2, 4))),
                 ('coef0', HyperParameter('INT', (0, 1))),
                 ('C', HyperParameter('FLOAT_EXP', (1e-05, 100000))),
                 ('gamma', HyperParameter('FLOAT_EXP', (1e-05, 100000)))]
            gridding: int. If a positive integer, controls the number of points
                on each axis of the grid. If 0, gridding does not occur.
        """
        self.tunables = tunables
        self.grid = gridding > 0
        self._best_score = -1 * float('inf')
        self._best_hyperparams = None

        if self.grid:
            self.grid_size = gridding
            self._define_grid()

        self.X_raw = None
        self.y_raw = []

        self.X = np.array([])
        self.y = np.array([])

    def _define_grid(self):
        """
        Define the range of possible values for each of the tunable
        hyperparameters.
        """
        self._grid_axes = []
        for _, param in self.tunables:
            self._grid_axes.append(param.get_grid_axis(self.grid_size))

    def _params_to_grid(self, params):
        """
        Fit a vector of continuous parameters to the grid. Each parameter is
        fitted to the grid point it is closest to.
        """
        # This list will be filled with hyperparameter vectors that have been
        # mapped from vectors of continuous values to vectors of indices
        # representing points on the grid.
        grid_points = []
        for i, val in enumerate(params):
            axis = self._grid_axes[i]
            # find the index of the grid point closest to the hyperparameter
            # vector
            idx = min(range(len(axis)), key=lambda i: abs(axis[i] - val))
            grid_points.append(idx)

        return np.array(grid_points)

    def _grid_to_params(self, grid_points):
        """
        Map a single point on the grid, represented by indices into each axis,
        to a continuous-valued parameter vector.
        """
        params = [self._grid_axes[i][p] for i, p in enumerate(grid_points)]
        return np.array(params)

    def fit(self, X, y):
        """
        Args:
            X: np.array of hyperparameters,
                shape = (n_samples, len(tunables))
            y: np.array of scores, shape = (n_samples,)
        """
        self.X = X
        self.y = y

    def _create_candidates(self, n=1000):
        """
        Generate a number of random hyperparameter vectors based on the
        specifications in self.tunables

        Args:
            n (optional): number of candidates to generate

        Returns:
            np.array of candidate hyperparameter vectors,
                shape = (n_samples, len(tunables))
        """

        # If using a grid, generate a list of previously unused grid points
        if self.grid:
            # convert numpy array to set of tuples of grid indices for easier
            # comparison
            past_vecs = set(tuple(self._params_to_grid(v)) for v in self.X)

            # if every point has been used before, gridding is done.
            num_points = self.grid_size ** len(self.tunables)
            if len(past_vecs) == num_points:
                return None

            # if fewer than n total points have yet to be seen, just return all
            # grid points not in past_vecs
            if num_points - len(past_vecs) <= n:
                # generate all possible points in the grid
                # FIXED BUG: indices = np.indices(self._grid_axes)
                indices = np.indices(len(a) for a in self._grid_axes)
                all_vecs = set(tuple(v) for v in
                               indices.T.reshape(-1, indices.shape[0]))
                vec_list = list(all_vecs - past_vecs)
            else:
                # generate n random vectors of grid-point indices
                vec_list = []
                for i in range(n):
                    # TODO: only choose from set of unused values
                    while True:
                        vec = np.random.randint(self.grid_size,
                                                size=len(self.tunables))
                        if tuple(vec) not in past_vecs:
                            break
                    vec_list.append(vec)

            # map the points back to continuous values and return
            return np.array([self._grid_to_params(v) for v in vec_list])

        # If not using a grid, generate a list of vectors where each parameter
        # is chosen uniformly at random
        else:
            # generate a matrix of random parameters, column by column.
            candidates = np.zeros((n, len(self.tunables)))
            for i, (k, param) in enumerate(self.tunables):
                lo, hi = param.range
                if param.is_integer:
                    column = np.random.randint(lo, hi + 1, size=n)
                else:
                    diff = hi - lo
                    column = lo + diff * np.random.rand(n)

                candidates[:, i] = column
                i += 1

            return candidates

    def predict(self, X):
        """
        Args:
            X: np.array of hyperparameters,
                shape = (n_samples, len(tunables))

        Returns:
            y: np.array of predicted scores, shape = (n_samples)
        """
        raise NotImplementedError(
            'predict() needs to be implemented by a subclass of Tuner.')

    def _acquire(self, predictions):
        """
        Acquisition function. Accepts a list of predicted values for candidate
        parameter sets, and returns the index of the best candidate.

        Args:
            predictions: np.array of predictions, corresponding to a set of
                proposed hyperparameter vectors. Each prediction may be a
                sequence with more than one value.

        Returns:
            idx: index of the selected hyperparameter vector
        """
        return np.argmax(predictions)

    def propose(self, n=1):
        """
        Use the trained model to propose a new set of parameters.
        Args:
            n (optional): number of candidates to propose

        Returns:
            proposal: dictionary of tunable name to proposed value.
            If called with n>1 then proposal is a list of dictionaries.
        """
        proposed_params = []

        for i in range(n):
            # generate a list of random candidate vectors. If self.grid == True
            # each candidate will be a vector that has not been used before.
            candidate_params = self._create_candidates()

            # create_candidates() returns None when every grid point
            # has been tried
            if candidate_params is None:
                return None

            # predict() returns a tuple of predicted values for each candidate
            predictions = self.predict(candidate_params)

            # acquire() evaluates the list of predictions, selects one,
            # and returns its index.
            idx = self._acquire(predictions)

            # inverse transform acquired hyperparameters
            # based on hyparameter type
            params = {}
            for i in range(candidate_params[idx, :].shape[0]):
                inverse_transformed = self.tunables[i][1].inverse_transform(
                    candidate_params[idx, i]
                )
                params[self.tunables[i][0]] = inverse_transformed
            proposed_params.append(params)

        return params if n == 1 else proposed_params

    def add(self, X, y):
        """
        Add data about known tunable hyperparameter configurations and scores.
        Refits model with all data.
        Args:
            X: dict or list of dicts of hyperparameter combinations.
                Keys may only be the name of a tunable, and the dictionary
                must contain values for all tunables.
            y: float or list of floats of scores of the hyperparameter
                combinations. Order of scores must match the order
                of the hyperparameter dictionaries that the scores corresponds

        """
        if isinstance(X, dict):
            X = [X]
            y = [y]

        # transform the list of dictionaries into a np array X_raw
        for i in range(len(X)):
            each = X[i]
            # update best score and hyperparameters
            if y[i] > self._best_score:
                self._best_score = y[i]
                self._best_hyperparams = X[i]

            vectorized = []
            for tunable in self.tunables:
                vectorized.append(each[tunable[0]])

            if self.X_raw is not None:
                self.X_raw = np.append(
                    self.X_raw,
                    np.array([vectorized], dtype=object),
                    axis=0,
                )

            else:
                self.X_raw = np.array([vectorized], dtype=object)

        self.y_raw = np.append(self.y_raw, y)

        # transforms each hyperparameter based on hyperparameter type
        x_transformed = np.array([], dtype=np.float64)
        if len(self.X_raw.shape) > 1 and self.X_raw.shape[1] > 0:
            x_transformed = self.tunables[0][1].fit_transform(
                self.X_raw[:, 0],
                self.y_raw,
            ).astype(float)

            for i in range(1, self.X_raw.shape[1]):
                transformed = self.tunables[i][1].fit_transform(
                    self.X_raw[:, i],
                    self.y_raw,
                ).astype(float)
                x_transformed = np.column_stack((x_transformed, transformed))

        self.fit(x_transformed, self.y_raw)
