import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(
        self,
        y_true: NDArray[np.float64],
        y_pred: NDArray[np.float64]
    ) -> float:

        # small value to prevent log(0)
        epsilon = 1e-7

        # slightly shift predictions so they are never exactly 0
        y_pred = y_pred + epsilon

        # calculate binary cross-entropy loss
        # if y_true = 1 -> uses log(y_pred)
        # if y_true = 0 -> uses log(1 - y_pred)
        loss = -np.mean(
            y_true * np.log(y_pred) +
            (1 - y_true) * np.log(1 - y_pred)
        )

        # round result to 4 decimal places
        return round(loss, 4)

    def categorical_cross_entropy(
        self,
        y_true: NDArray[np.float64],
        y_pred: NDArray[np.float64]
    ) -> float:

        # small value to prevent log(0)
        epsilon = 1e-7

        # slightly shift predictions so they are never exactly 0
        y_pred = y_pred + epsilon

        # multiply one-hot labels with log probabilities
        # this keeps only the probability of the correct class
        loss = -np.mean(
            np.sum(y_true * np.log(y_pred), axis=1)
        )

        # round result to 4 decimal places
        return round(loss, 4)