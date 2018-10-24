import numpy as np

from ..execution.batch import Batch
from .sampling import LogSampler
from ..models.linear import LinearModel
from ..models.hill import HillModel
from ..models.twostate import TwoStateModel


class Sweep(Batch):
    """
    Class defines a parameter sweep of a given model.

    Attributes:

        base (np.ndarray[float]) - base parameter values

        delta (float or np.ndarray[float]) - log-deviations about base

    Inherited attributes:

        path (str) - path to batch directory

        parameters (np.ndarray[float]) - sampled parameter values

        simulation_paths (dict) - paths to simulation directories

        sim_kw (dict) - keyword arguments for simulation

        results (dict) - {simulation_id: results_dict} pairs

    Properties:

        N (int) - number of samples in parameter space

    """

    def __init__(self, base, delta=0.5, num_samples=1000):
        """
        Instantiate parameter sweep.

        Args:

            base (np.ndarray[float]) - base parameter values

            delta (float or np.ndarray[float]) - log-deviations about base

            num_samples (int) - number of samples in parameter space

        """

        self.base = base
        self.delta = delta

        # sample parameter space
        sampler = LogSampler(base-delta, base+delta)
        parameters = sampler.sample(num_samples)

        # instantiate batch job
        super().__init__(parameters=parameters)

    # def aggregate(self):
    #     """
    #     Aggregate results from each simulation.
    #     """

    #     sim.comparisons


    #     def get_error(sim):

    #         sim.comparisons




    #     comparisons[condition] = comparison



    #     self.apply(func)


class LinearSweep(Sweep):

    """
    Parameter sweep for linear model. Parameters are:

        0: activation rate constant
        1: transcription rate constant
        2: translation rate constant
        3: deactivation rate constant
        4: mrna degradation rate constant
        5: protein degradation rate constant
        6: transcriptional feedback strength
        7: post-transcriptional feedback strength
        8: post-translational feedback strength

    """

    def __init__(self, base=None, delta=0.5, num_samples=1000):
        """
        Instantiate parameter sweep of a linear model.

        Args:

            base (np.ndarray[float]) - base parameter values

            delta (float or np.ndarray[float]) - log-deviations about base

            num_samples (int) - number of samples in parameter space

        """

        # define parameter ranges, log10(val)
        if base is None:
            base = np.array([0, 0, 0, 0, -2, -3, -4.5, -4.5, -4.5])

        # call parent instantiation
        super().__init__(base, delta, num_samples)

    @staticmethod
    def build_model(parameters):
        """
        Returns a model instance defined by the provided parameters.

        Args:

            parameters (np.ndarray[float]) - model parameters

        Returns:

            model (LinearModel)

        """

        # extract parameters
        k0, k1, k2, g0, g1, g2, eta0, eta1, eta2 = parameters

        # instantiate base model
        model = LinearModel(k0=k0, k1=k1, k2=k2, g0=g0, g1=g1, g2=g2)

        # add feedback (two equivalent sets)
        model.add_feedback(eta0, eta1, eta2, perturbed=False)
        model.add_feedback(eta0, eta1, eta2, perturbed=True)

        return model


class HillSweep(Sweep):

    """
    Parameter sweep of a hill model. Parameters are:

        0: transcription hill coefficient
        1: transcription rate constant
        2: translation rate constant
        3: mrna degradation rate constant
        4: protein degradation rate constant
        5: repressor michaelis constant
        6: repressor hill coefficient
        7: post-transcriptional feedback strength
        8: post-translational feedback strength

    """

    def __init__(self, base=None, delta=0.5, num_samples=1000):
        """
        Instantiate parameter sweep of a Hill model.

        Args:

            base (np.ndarray[float]) - base parameter values

            delta (float or np.ndarray[float]) - log-deviations about base

            num_samples (int) - number of samples in parameter space

        """

        # define parameter ranges, log10(val)
        if base is None:
            base = np.array([0, 0, 0, -2, -3, -4, 0, -5, -4])

        # call parent instantiation
        super().__init__(base, delta, num_samples)

    @staticmethod
    def build_model(parameters):
        """
        Returns a model instance defined by the provided parameters.

        Args:

            parameters (np.ndarray[float]) - model parameters

        Returns:

            model (HillModel)

        """

        # extract parameters
        n, k1, k2, g1, g2, k_m, r_n, eta1, eta2 = parameters

        # instantiate base model
        model = HillModel(k1=k1, k_m=.5, n=n, k2=k2, g1=g1, g2=g2)

        # add feedback (two equivalent sets)
        model.add_feedback(k_m, r_n, eta1, eta2, perturbed=False)
        model.add_feedback(k_m, r_n, eta1, eta2, perturbed=True)

        return model


class TwoStateSweep(Sweep):

    """
    Parameter sweep of a twostate model. Parameters are:

        0: activation rate constant
        1: transcription rate constant
        2: translation rate constant
        3: deactivation rate constant
        4: mrna degradation rate constant
        5: protein degradation rate constant
        6: transcriptional feedback strength
        7: post-transcriptional feedback strength
        8: post-translational feedback strength

    """

    def __init__(self, base=None, delta=0.5, num_samples=1000):
        """
        Instantiate parameter sweep of a twostate model.

        Args:

            base (np.ndarray[float]) - base parameter values

            delta (float or np.ndarray[float]) - log-deviations about base

            num_samples (int) - number of samples in parameter space

        """

        # define parameter ranges, log10(val)
        if base is None:
            base = np.array([0, 0, 0, -1, -2, -3, -4, -4.5, -4])

        # call parent instantiation
        super().__init__(base, delta, num_samples)

    @staticmethod
    def build_model(parameters):
        """
        Returns a model instance defined by the provided parameters.

        Args:

            parameters (np.ndarray[float]) - model parameters

        Returns:

            model (LinearModel)

        """

        # extract parameters
        k0, k1, k2, g0, g1, g2, eta0, eta1, eta2 = parameters

        # instantiate base model
        model = TwoStateModel(k0=k0, k1=k1, k2=k2, g0=g0, g1=g1, g2=g2)

        # add feedback (two equivalent sets)
        model.add_feedback(eta0, eta1, eta2, perturbed=False)
        model.add_feedback(eta0, eta1, eta2, perturbed=True)

        return model
