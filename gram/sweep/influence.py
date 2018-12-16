import numpy as np
from copy import deepcopy
from ..execution.batch import Batch
from .sampling import DenseLinearSampler, DenseLogSampler
from ..models.simple import SimpleModel
from .sweep import Sweep


class RepressorInfluenceSweep(Sweep):

    """
    2D parameter sweep for simple birth-death model. Parameters are:

        0: degradation strength
        1: relative influence of lost repressor

    """

    def __init__(self, base=None, delta=None, num_samples=51):
        """
        Instantiate parameter sweep of a simple model.

        Args:

            base (np.ndarray[float]) - base parameter values (log scale)

            delta (float or np.ndarray[float]) - deviations about base

            num_samples (int) - number of samples per parameter

        """

        # define parameter ranges
        if base is None:
            base = np.array([-10, 0])

        if delta is None:
            delta = np.array([2, 5])

        self.base = base
        self.delta = delta
        self.labels = ('Repressor Strength', 'Relative Loss')
        self.results = None

        # sample parameter space
        sampler = DenseLogSampler(base-delta, base+delta)
        parameters = sampler.sample(num_samples)

        # instantiate batch job
        Batch.__init__(self, parameters=parameters)

        # set run script
        self.script_name = 'run_simple_batch.py'

    @staticmethod
    def build_model(parameters):
        """
        Returns a model instance defined by the provided parameters.

        Args:

            parameters (np.ndarray[float]) - model parameters

        Returns:

            model (LinearModel)

        """

        eta, relative_influence = parameters

        k, g = 1, eta/relative_influence

        # extract parameters
        lambda_k, lambda_g = 1., 1. #0.5

        # instantiate base model
        model = SimpleModel(k=k, g=g, lambda_g=lambda_g, lambda_k=lambda_k)

        # add feedback (second perturbation-sensitive decay reaction)
        fb_rxn = deepcopy(model.reactions[0])
        assert fb_rxn.__class__.__name__ == 'MassAction', 'Wrong reaction.'
        fb_rxn.k[0] = eta
        fb_rxn.labels['perturbed'] = True
        model.reactions.append(fb_rxn)

        return model


class PromoterInfluenceSweep(RepressorInfluenceSweep):

    """
    2D parameter sweep for simple birth-death model. Parameters are:

        0: promoter strength
        1: relative influence of lost promoter

    """

    def __init__(self, base=None, delta=None, num_samples=51):
        """
        Instantiate parameter sweep of a simple model.

        Args:

            base (np.ndarray[float]) - base parameter values (log scale)

            delta (float or np.ndarray[float]) - deviations about base

            num_samples (int) - number of samples per parameter

        """

        if base is None:
            base = np.array([0, 0])

        super().__init__(base, delta, num_samples)
        self.labels = ('Promoter Strength', 'Relative Loss')

    @staticmethod
    def build_model(parameters):
        """
        Returns a model instance defined by the provided parameters.

        Args:

            parameters (np.ndarray[float]) - model parameters

        Returns:

            model (LinearModel)

        """


        k_mutant, relative_influence = parameters

        k, g = k_mutant/relative_influence, 0.001

        # extract parameters
        lambda_k, lambda_g = 1., 1. #0.5

        # instantiate base model
        model = SimpleModel(k=k, g=g, lambda_g=lambda_g, lambda_k=lambda_k)

        # add promoter (second perturbation-sensitive activation reaction)
        act_rxn = deepcopy(model.reactions[-1])
        assert act_rxn.__class__.__name__ == 'MassAction', 'Wrong reaction.'
        act_rxn.k[0] = k_mutant
        act_rxn.labels['perturbed'] = True
        model.reactions.append(act_rxn)

        return model
