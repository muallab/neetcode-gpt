import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # Step 1: subtract the maximum value for numerical stability
        z_stable = z - np.max(z)

        # Step 2: apply exponent to each value
        exp_values = np.exp(z_stable)

        # Step 3: divide by the sum to get probabilities
        probabilities = exp_values / np.sum(exp_values)

        # Step 4: round to 4 decimal places
        return np.round(probabilities, 4)