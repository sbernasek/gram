import numpy as np
import pandas as pd
from os.path import join, exists

from ..execution.batch import Batch
from .sampling import LogSampler, LinearSampler
from ..models.linear import LinearModel
from ..models.hill import HillModel
from ..models.twostate import TwoStateModel
from ..models.simple import SimpleModel
from .figure import SweepFigure


class Sweep(Batch):
    """
    Class defines a parameter sweep of a given model.

    Attributes:

        base (np.ndarray[float]) - base parameter values

        delta (float or np.ndarray[float]) - log-deviations about base

        labels (list of str) - labels for each parameter

        results (pd.DataFrame) - results with (condition, mode) multiindex

    Inherited attributes:

        path (str) - path to batch directory

        run_script (str) - path to script for running batch

        parameters (np.ndarray[float]) - sampled parameter values

        simulation_paths (dict) - paths to simulation directories

        sim_kw (dict) - keyword arguments for simulation

    Properties:

        N (int) - number of samples in parameter space

    """

    def __init__(self,
                 base,
                 delta=0.5,
                 num_samples=1000,
                 labels=None,
                 pad=.1,
                 logbase=10):
        """
        Instantiate parameter sweep.

        Args:

            base (np.ndarray[float]) - base parameter values

            delta (float or np.ndarray[float]) - log-deviations about base

            num_samples (int) - number of samples in parameter space

            labels (list of str) - labels for each parameter

            pad (float) - extra padding added to delta

            logbase (float) - logarithmic basis for sampling

        """

        self.base = base
        self.delta = delta
        self.pad = pad
        self.labels = labels
        self.results = None

        # sample parameter space
        sampler = LogSampler(base-delta-pad, base+delta+pad, base=logbase)
        parameters = sampler.sample(num_samples)

        # instantiate batch job
        super().__init__(parameters=parameters)

    @staticmethod
    def load(path):
        """ Load sweep from target <path>. """

        sweep = Batch.load(path)

        # load results
        if exists(join(path, 'data.hdf')):
            sweep.results = pd.read_hdf(join(path, 'data.hdf'), 'results')
            sweep.completed = pd.read_hdf(join(path, 'data.hdf'), 'completed')

        return sweep

    def save(self):
        """ Save sweep data. """
        if self.results is not None:
            p = join(self.path, 'data.hdf')
            self.results.to_hdf(p, 'results', mode='a')
            self.completed.to_hdf(p, 'completed', mode='a')

    @staticmethod
    def parse_simulation(simulation):
        """ Returns over, under, and total error from <simulation>. """
        if simulation.comparisons is None:
            return False
        else:
            errors = {}
            for condition, comparison in simulation.comparisons.items():
                errors[(condition, 'above')] = comparison.above
                errors[(condition, 'below')] = comparison.below
                errors[(condition, 'error')] = comparison.error
            return errors

    @property
    def percent_complete(self):
        """ Fraction of simulations that ran to completion. """
        return self.completed.values.sum()/self.N

    def aggregate(self):
        """
        Aggregate results from each completed simulation and compile them into a dataframe.
        """

        # parse simulation results
        results = [self.parse_simulation(sim) for sim in self]

        # store indices of completed simulations
        index = np.arange(self.N)
        self.completed = pd.DataFrame([x!=False for x in results], index=index)

        # compile results dataframe
        error_dicts = [x for x in results if x != False]
        self.results = pd.DataFrame.from_dict(error_dicts, orient='columns')
        self.results.columns = pd.MultiIndex.from_tuples(self.results.columns)

    def slice_by_mode(self, mode='error'):
        """ Returns all results for a specified <mode>. """
        return self.results.swaplevel(axis=1)[mode]

    def slice_by_condition(self, condition='normal'):
        """ Returns all results for a specified <condition>. """
        return self.results[condition]

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

        return SweepFigure(self.parameters,
                           results,
                           labels=self.labels,
                           base=self.base,
                           delta=self.delta,
                           **kwargs)


class SimpleSweep(Sweep):

    """
    Parameter sweep for simple model. Parameters are:

        0: synthesis rate constant
        1: decay rate constant
        2: feedback strength

    """

    def __init__(self, base=None, delta=0.5, num_samples=2500):
        """
        Instantiate parameter sweep of a simple model.

        Args:

            base (np.ndarray[float]) - base parameter values

            delta (float or np.ndarray[float]) - log-deviations about base

            num_samples (int) - number of samples in parameter space

        """

        # define parameter ranges, log10(val)
        if base is None:
            base = np.array([0, -3, -3])

        # define parameter labels
        labels = ('k', '\gamma', '\eta')

        # call parent instantiation
        super().__init__(base, delta, num_samples, labels=labels)

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
        k, g, eta = parameters

        # instantiate base model
        model = SimpleModel(k=k, g=g)

        # add feedback (two equivalent sets)
        model.add_feedback(eta, perturbed=True)

        return model


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
            base = np.array([0, 0, 0, 0, -2, -3, -4, -4, -4])

        # define parameter labels
        labels = ('k_0', 'k_1', 'k_2',
                  '\gamma_0', '\gamma_1', '\gamma_2',
                  '\eta_0', '\eta_1', '\eta_2')

        # call parent instantiation
        super().__init__(base, delta, num_samples, labels=labels)

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
            base = np.array([0, 0, 0, -2, -3, 4, 0, -5, -4])

        # define parameter labels
        labels = ('H', 'k_R', 'k_P',
                  '\gamma_R', '\gamma_P',
                  'K_r', 'H_r',
                  '\eta_R', '\eta_P')

        # call parent instantiation
        super().__init__(base, delta, num_samples, labels=labels)

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

        # define parameter labels
        labels = ('k_G', 'k_R', 'k_P',
                  '\gamma_G', '\gamma_R', '\gamma_P',
                  '\eta_G', '\eta_R', '\eta_P')

        # call parent instantiation
        super().__init__(base, delta, num_samples, labels=labels)

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


class SimpleDependenceSweep(Sweep):

    """
    Parameter sweep for simple model. Parameters are:

        0: synthesis growth dependence
        1: degradation growth dependence
        2: feedback strength growth dependence

    """

    def __init__(self, base=None, delta=2, pad=0.2, num_samples=2500):
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
            base = np.array([0, 0, 0])

        self.base = base
        self.delta = delta
        self.pad = pad
        self.labels = ('k', '\gamma', '\eta')
        self.results = None

        # sample parameter space
        sampler = LinearSampler(base-delta-pad, base+delta+pad)
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
