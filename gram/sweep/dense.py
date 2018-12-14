import numpy as np
from copy import deepcopy
from ..execution.batch import Batch
from ..models.simple import SimpleModel
from .figure import LinearSweepFigure
from .sweep import Sweep
from .sampling import DenseLinearSampler


class SimpleDense2D(Sweep):

    """
    2D parameter sweep for simple birth-death model. Parameters are:

        0: synthesis growth dependence
        1: degradation growth dependence

    """

    def __init__(self, base=None, delta=0, num_samples=51):
        """
        Instantiate parameter sweep of a simple model.

        Args:

            base (np.ndarray[float]) - base parameter values

            delta (float or np.ndarray[float]) - deviations about base

            num_samples (int) - number of samples per parameter

        """

        # define parameter ranges, log10(val)
        if base is None:
            base = np.array([2, 2])

        self.base = base
        self.delta = delta
        self.labels = ('Synthesis', 'Decay')
        self.results = None

        # sample parameter space
        sampler = DenseLinearSampler(base-delta, base+delta)
        parameters = sampler.sample(num_samples)

        # instantiate batch job
        Batch.__init__(self, parameters=parameters)

        # set run script
        self.script_name = 'run_dependence.py'

    @staticmethod
    def build_model(parameters):
        """
        Returns a model instance defined by the provided parameters.

        Args:

            parameters (np.ndarray[float]) - model parameters

        Returns:

            model (LinearModel)

        """

        k, g, eta = 1, 0.001, 0.001

        # extract parameters
        lambda_k, lambda_g = parameters

        # instantiate base model
        model = SimpleModel(k=k, g=g, lambda_g=lambda_g, lambda_k=lambda_k)

        # add feedback (second perturbation-sensitive decay reaction)
        decay_rxn = deepcopy(model.reactions[0])
        assert decay_rxn.__class__.__name__ == 'MassAction', 'Wrong reaction.'
        decay_rxn.labels['perturbed'] = True
        model.reactions.append(decay_rxn)

        return model


class SimpleDependenceSweep(Sweep):

    """
    Parameter sweep for simple model. Parameters are:

        0: synthesis growth dependence
        1: degradation growth dependence
        2: feedback strength growth dependence

    """

    def __init__(self, base=None, delta=1, pad=0., num_samples=11):
        """
        Instantiate parameter sweep of a simple model.

        Args:

            base (np.ndarray[float]) - base parameter values

            delta (float or np.ndarray[float]) - log-deviations about base

            pad (float) - extra padding added to delta

            num_samples (int) - number of samples in parameter space

        """

        # define parameter ranges, log10(val)
        if base is None:
            base = np.array([1, 1, 1])

        self.base = base
        self.delta = delta
        self.pad = pad
        self.labels = ('Synthesis', 'Decay', 'Feedback')
        self.results = None

        # sample parameter space
        sampler = DenseLinearSampler(base-delta, base+delta)
        parameters = sampler.sample(num_samples)

        # instantiate batch job
        Batch.__init__(self, parameters=parameters)

        # set run script
        self.script_name = 'run_dependence.py'

    @staticmethod
    def build_model(parameters):
        """
        Returns a model instance defined by the provided parameters.

        Args:

            parameters (np.ndarray[float]) - model parameters

        Returns:

            model (LinearModel)

        """

        k, g, eta = 1, 0.001, 0.001

        # extract parameters
        lambda_k, lambda_g, lambda_eta = parameters

        # instantiate base model
        model = SimpleModel(k=k, g=g, lambda_g=lambda_g, lambda_k=lambda_k)

        # add feedback (two equivalent sets)
        model.add_feedback(eta, perturbed=True, lambda_eta=lambda_eta)

        return model

    def build_figure(self,
                     condition='normal',
                     mode='error',
                     relative=False,
                     **kwargs):
        """
        Returns parameter sweep visualization.

        Args:

            condition (str or tuple) - environmental condition

            mode (str) - comparison metric

            relative (bool) - if True, computes difference relative to normal

            kwargs: keyword arguments for SweepFigure

        """

        # evaluate results
        results = self.results.loc[:, (condition, mode)]
        if relative:
            results = results - self.results.loc[:, ('normal', mode)]

        return LinearSweepFigure(
            self.parameters,
            results,
            labels=self.labels,
            base=self.base,
            delta=self.delta,
            **kwargs)
