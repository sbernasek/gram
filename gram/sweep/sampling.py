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
    Class for uniformly sampling a parameter space on a log scale.

    Attributes:

        low (np.ndarray[float]) - log10 lower bound for each parameter

        high (np.ndarray[float]) - log10 upper bound for each parameter

        base (float) - basis for logarithmic sampling

    Inherited attributes:

        dim (int) - dimensionality of sampled space

    """
    def __init__(self, *args, base=10):
        """
        Instantiate sobol sampler.

        Args:

            low (np.ndarray[float]) - lower bound for each parameter

            high (np.ndarray[float]) - upper bound for each parameter

            base (float) - basis for logarithmic sampling

        """
        self.base = base
        super().__init__(*args)

    def sample(self, N):
        """ Returns an array of <N> sobol samples. """
        return self.base**super().sample(N)
