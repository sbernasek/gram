from os.path import join, isdir
from os import mkdir
from collections import OrderedDict
from time import time
import matplotlib.pyplot as plt

from genessa.timeseries.base import TimeSeries
from .perturbation import PerturbationSimulation


class ConditionSimulation(PerturbationSimulation):
    """
    Numerical simulations of a single gene expression pulse before and after a genetic perturbations under a range of environmental conditions.

    Attributes:

        conditions (array like) - simulation conditions

        dynamics (dict) - {condition: (before, after)} pairs in which before and after are simulation trajectories stored as TimeSeries objects

        comparisons (dict) - {condition: Comparison} pairs

        saveall (bool) - if True, dynamics were saved

        horizon (float) - duration of comparison

        deviations (bool) - if True, compare deviations from initial value

        seed (int) - seed used for random number generator

        runtime (float) - total runtime

        complete (bool) - flag for completing a simulation

    Inherited Attributes:

        cell (Cell derivative)

        mutant (Cell derivative) - cell with perturbation applied

        pulse_start (float) - pulse onset time

        pulse_duration (float) - pulse duration under normal conditions

        pulse_baseline (float) - basal signal level

        pulse_magnitude (float) - magnitude of pulse (increase over baseline)

        pulse_sensitive (bool) - indicates whether pulse duration depends upon environmental conditions

        simulation_duration (float) - simulation duration

        dt (float) - sampling interval

        timescale (float) - time scaling factor

    """

    def __init__(self, cell, **kwargs):
        """
        Instantiate environmental comparison simulation.

        Args:

            cell (Cell derivative)

        Keyword Arguments:

            pulse_start (float) - pulse onset time

            pulse_duration (float) - pulse duration under normal conditions

            pulse_baseline (float) - basal signal level

            pulse_magnitude (float) - magnitude of pulse

            pulse_sensitive (bool) - if True, pulse duration depends upon environmental conditions

            simulation_duration (float) - simulation duration

            dt (float) - sampling interval

            timescale (float) - time scaling factor

        """

        super().__init__(cell, **kwargs)

        # initialize conditions, dynamics, and comparisons dictionaries
        self.conditions = []
        self.dynamics = None
        self.comparisons = None

        # initialize comparison attributes
        self.horizon = None
        self.deviations = None

        # initialize runtime and completion flag
        self.runtime = -1
        self.complete = False

        # set condition names
        self.condition_names = dict(normal='Normal',
                                  diabetic='Reduced Metabolism',
                                  minute='Reduced Translation',
                                  half_growth='Halved Growth Rate')

    def __getstate__(self):
        """ Returns all attributes except simulation trajectories. """
        return {k: v for k, v in self.__dict__.items() if k != 'dynamics'}

    @property
    def N(self):
        """ Number of environmental conditions. """
        return len(self.comparisons)

    @classmethod
    def load(cls, path):
        """
        Load simulation from file.

        Args:

            path (str) - file path

        Returns:

            simulation (ConditionSimulation)

        """

        # load serialized simulation instance
        simulation = super(cls, cls).load(join(path, 'simulation.pkl'))

        # load simulation trajectories (if available)
        if simulation.saveall:
            cls.load_trajectories(path, simulation)

        return simulation

    @staticmethod
    def load_trajectories(path, simulation):
        """
        Load simulation trajectories from file.

        Args:

            simulation (ConditionSimulation) - file path

        """
        # load simulation trajectories (if available)
        simulation.dynamics = OrderedDict()
        for condition in simulation.conditions:

            # check that directory exists
            subdir = join(path, 'dynamics', condition)
            if not isdir(subdir):
                continue

             # load simulation trajectories for control
            control_dir = join(subdir, 'control')
            if isdir(control_dir):
                before = TimeSeries.load(control_dir)

            # load simulation trajectories for perturbation
            perturbation_dir = join(subdir, 'perturbation')
            if isdir(perturbation_dir):
                after = TimeSeries.load(perturbation_dir)

            # store simulation trajectories for current condition
            simulation.dynamics[condition] = (before, after)

        # set simulation trajectories for comparison objects (if available)
        for condition, comparison in simulation.comparisons.items():

            # get simulation trajectories for comparison condition
            before = simulation.dynamics[condition][0]
            after = simulation.dynamics[condition][1]

            # transform to deviations
            if simulation.deviations == True:
                before = before.get_deviations(values='final')
                after = after.get_deviations(values='final')

            # crop timeseries
            if simulation.horizon is not None:
                start = simulation.pulse_start / simulation.timescale
                stop = start + simulation.horizon
                before = before.crop(start, stop)
                after = after.crop(start, stop)

            # if comparison uses a different type, cast the timeseries
            if comparison.tstype != TimeSeries:
                tskwargs = comparison.tskwargs
                before = comparison.tstype.from_timeseries(before, **tskwargs)
                after = comparison.tstype.from_timeseries(after, **tskwargs)

            # set trajectories for comparison
            comparison.reference = before
            comparison.compared = after

    def save(self, path, saveall=False):
        """
        Save simulation to file. Simulations are saved as serialized pickle objects. TimeSeries data may optionally be saved as numpy arrays.

        Args:

            path (str) - save destination

            saveall (bool) - if True, save timeseries data

        """

        if saveall:

            # make a directory for simulation dynamics
            dynamics_dir = join(path, 'dynamics')
            if not isdir(dynamics_dir):
                mkdir(dynamics_dir)

            # save dynamics for each condition
            for condition, (before, after) in self.dynamics.items():

                # make a directory
                subdir = join(dynamics_dir, condition)
                if not isdir(subdir):
                    mkdir(subdir)

                # save simulation trajectories
                before.save(join(subdir, 'control'))
                after.save(join(subdir, 'perturbation'))

        # save serialized object
        self.saveall = saveall
        super().save(join(path, 'simulation.pkl'))

    def simulate(self,
                 N=1000,
                 conditions=None,
                 seed=None,
                 inplace=True,
                 debug=False):
        """
        Run perturbation simulation for each environmental condition.

        Args:

            N (int) - number of independent simulation trajectories

            conditions (array like) - simulation conditions

            seed (int) - seed for random number generator

            inplace (bool) - if True, store simulation trajectories

            debug (bool) - if True, use debugging mode

        Returns:

            dynamics (dict) - {condition: (before, after)} pairs in which before and after are simulation trajectories stored as TimeSeries objects

        """

        # set simulation conditions
        if conditions is None:
            conditions = ('normal', 'diabetic', 'minute')
        self.conditions = conditions

        # run simulations
        kwargs = dict(N=N, seed=seed, debug=debug)
        dynamics = OrderedDict()
        for condition in conditions:
            before, after = super().simulate(condition, **kwargs)
            dynamics[condition] = (before, after)

        # set/return dynamics
        if inplace:
            self.dynamics = dynamics
        else:
            return dynamics

    def compare(self,
                mode=None,
                horizon=0,
                deviations=False,
                inplace=True,
                **kwargs):
        """
        Evaluate comparison for each environmental condition.

        Args:

            mode (str) - comparison type, options are:
                empirical: fraction of trajectories below/above reference
                area: fraction of confidence band area below/above reference
                cdf: fraction of gaussian model below/above reference
                threshold: fraction of gaussian model above threshold

            horizon (float) - duration of comparison, 0 if unlimited

            deviations (bool) - if True, compare deviations from initial value

            inplace (bool) - if True, store simulation trajectories

            kwargs: keyword arguments for comparison

        Returns:

            comparisons (dict) - {condition: Comparison} pairs

        """

        # define keyword arguments for comparison
        kw = dict(mode=mode, horizon=horizon, deviations=deviations, **kwargs)

        # run comparisons
        comparisons = OrderedDict()
        for condition, (reference, compared) in self.dynamics.items():

            # evaluate comparison
            comparison = super().compare(reference, compared, **kw)
            comparisons[condition] = comparison

        # set/return dynamics
        if inplace:
            self.horizon = horizon
            self.deviations = deviations
            self.comparisons = comparisons
        else:
            return comparisons

    def run(self, skwargs={}, ckwargs={}):
        """
        Run simulations and evaluate comparisons.

        Args:

          skwargs (dict) - keyword arguments for simulation

          ckwargs (dict) - keyword arguments for comparison

        """

        start_time = time()

        # generate seed for random number generator
        if 'seed' in skwargs.keys():
            seed = skwargs.pop('seed')
        else:
            seed = int(start_time)

        # run simulation and comparison
        self.simulate(seed=seed, **skwargs)
        self.compare(**ckwargs)

        # store seed and runtime
        self.runtime = time() - start_time
        self.seed = seed
        self.complete = True

    def plot_comparison(self, trajectories=False, axes=None):
        """
        Visualize comparison for each environmental condition.

        Args:

            trajectories (bool) - if True, plot individual trajectories

            axes (tuple) - matplotlib.axes.AxesSubplot for each condition

        """

        # create axes if none were provided
        if axes is None:
            ncols = self.N
            figsize=(ncols*2.5, 2)
            fig, axes = plt.subplots(1, ncols, sharey=True, figsize=figsize)

        # visualize comparison under each condition
        for i, (condition, comparison) in enumerate(self.comparisons.items()):

            if trajectories:
                comparison.plot_outlying_trajectories(ax=axes[i])
            else:
                comparison.shade_outlying_areas(ax=axes[i])
            axes[i].set_title(self.condition_names[condition])

        # display error metrics on plot
        for i, comparison in enumerate(self.comparisons.values()):
            comparison.display_metrics(axes[i])

        axes[0].set_ylabel('Protein level')

        plt.tight_layout()

    def plot_dynamics(self, dim=-1, trajectories=False, axes=None):
        """
        Visualize full expression dynamics for each environmental condition.

        Args:

            dim (int) - state space dimension to be visualized

            trajectories (bool) - if True, plot individual trajectories

            axes (tuple) - matplotlib.axes.AxesSubplot for each condition

        """

        # create axes if none were provided
        if axes is None:
            ncols = self.N
            figsize=(ncols*2.5, 2)
            fig, axes = plt.subplots(1, ncols, sharey=True, figsize=figsize)

        # visualize comparison under each condition
        for i, (condition, dynamics) in enumerate(self.dynamics.items()):
            before, after = dynamics

            # plot dynamics
            if trajectories:
                self.plot_trajectories(axes[i], before, dim, c='k')
                self.plot_trajectories(axes[i], after, dim, c='r')
            else:
                self.plot_confidence_band(axes[i], before, dim, c='k')
                self.plot_confidence_band(axes[i], after, dim, c='r')

            # add plot title
            axes[i].set_title(self.condition_names[condition])

        axes[0].set_ylabel('Protein level')

        plt.tight_layout()

    @staticmethod
    def plot_confidence_band(ax, timeseries, dim, c='k', alpha=0.2, line=True):
        """ Plots confidence band for <dim> of <timeseries> on <ax>. """
        lb, ub = timeseries.lower[dim], timeseries.upper[dim]
        ax.fill_between(timeseries.t, lb, ub, color=c, alpha=alpha)
        if line:
            ax.plot(timeseries.t, lb, '-k')
            ax.plot(timeseries.t, ub, '-k')

    @staticmethod
    def plot_trajectories(ax, timeseries, dim, c='k', alpha=0.2, lw=0.1):
        """ Plots trajectories for <dim> of <timeseries> on <ax>. """
        for trajectory in timeseries.states[:, dim, :]:
            ax.plot(timeseries.t, trajectory, color=c, alpha=alpha, lw=lw)
