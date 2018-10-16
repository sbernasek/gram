import numpy as np
from sobol import i4_sobol


class SobolSampler:
    """
    Class for sobol sampling a parameter space.

    Attributes:

        dim (int) - dimensionality of sampled space

    """

    def __init__(self, dim):
        """
        Instantiate sobol sampler.

        Args:

            dim (int) - dimensionality of sampled space


        """

        self.dim = dim

    def sample(self, N):
        """ Returns an array of <N> sobol samples. """
        return np.array([i4_sobol(self.dim, i)[0] for i in range(N)])


class LinearSampler(SobolSampler):
    """
    Class for sobol sampling a parameter space on a linear scale.

    Attributes:

        low (np.ndarray[float]) - lower bound for each parameter

        high (np.ndarray[float]) - upper bound for each parameter

    Inherited attributes:

        dim (int) - dimensionality of sampled space

    """

    def __init__(self, low, high):
        """
        Instantiate sobol sampler.

        Args:

            low (np.ndarray[float]) - lower bound for each parameter

            high (np.ndarray[float]) - upper bound for each parameter

        """
        super().__init__(len(low))
        self.low = low
        self.high = high

    def sample(self, N):
        """ Returns an array of <N> sobol samples. """
        return self.low+(self.high-self.low)*super().sample(N)


class LogSampler(LinearSampler):
    """
    Class for log sampling a parameter space on a linear scale.

    Attributes:

        low (np.ndarray[float]) - log10 lower bound for each parameter

        high (np.ndarray[float]) - log10 upper bound for each parameter

    Inherited attributes:

        dim (int) - dimensionality of sampled space

    """
    def sample(self, N):
        """ Returns an array of <N> sobol samples. """
        return 10**super().sample(N)
