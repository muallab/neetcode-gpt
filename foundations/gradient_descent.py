class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Start with an initial value
        x = init

        # Perform gradient descent updates
        for _ in range(iterations):

            # Compute derivative f'(x) = 2x
            gradient = 2 * x

            # Update x
            x = x - learning_rate * gradient
            
        return round(x,5)



        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
