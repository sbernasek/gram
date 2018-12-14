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


class SobolLinearSampler(SobolSampler):
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


class SobolLogSampler(SobolLinearSampler):
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


class DenseLinearSampler:
    """
    Class for dense sampling a parameter space on a linear scale.

    Attributes:

        dim (int) - dimensionality of sampled space

        low (np.ndarray[float]) - lower bound for each parameter

        high (np.ndarray[float]) - upper bound for each parameter

    """

    def __init__(self, low, high):
        """
        Instantiate sobol sampler.

        Args:

            low (np.ndarray[float]) - lower bound for each parameter

            high (np.ndarray[float]) - upper bound for each parameter

        """
        self.dim = len(low)
        self.low = low
        self.high = high

    def sample(self, density):
        """
        Returns an array of dense samples.

        Args:

            density (int) - number of samples per dimension

        """
        bounds = zip(self.low, self.high)
        points = [np.linspace(l, h, density) for l,h in bounds]
        grids = np.meshgrid(*points, indexing='ij')
        return np.stack([g.ravel() for g in grids]).T


class DenseLogSampler(DenseLinearSampler):
    """
    Class for dense sampling a parameter space on a log scale.

    Attributes:

        dim (int) - dimensionality of sampled space

        low (np.ndarray[float]) - lower bound for each parameter

        high (np.ndarray[float]) - upper bound for each parameter

        base (float) - basis for logarithmic sampling

    """

    def __init__(self, low, high, base=2):
        """
        Instantiate sobol sampler.

        Args:

            low (np.ndarray[float]) - lower log-bound for each parameter

            high (np.ndarray[float]) - upper log-bound for each parameter

            base (float) - basis for logarithmic sampling

        """
        super().__init__(low, high)
        self.base = base

    def sample(self, density):
        """ Returns an array of <N> sobol samples. """
        return self.base**super().sample(density)
