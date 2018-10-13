from os.path import join, isdir
from os import mkdir
from collections import OrderedDict
import matplotlib.pyplot as plt

from .perturbation import PerturbationSimulation


class EnvironmentSimulation(PerturbationSimulation):
    """
    Numerical simulations of a single gene expression pulse before and after a genetic perturbations under a range of environmental conditions.

    Attributes:

        comparisons (dict) - {condition: AreaComparison} pairs

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

    def __init__(self, cell, conditions=None, **kwargs):
        """
        Instantiate environmental comparison simulation.

        Args:

            cell (Cell derivative)

            conditions (array like) - conditions to be compared

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

        # initialize comparisons
        if conditions is None:
            conditions = ('normal', 'diabetic', 'minute')
        self.comparisons = OrderedDict([(c, None) for c in conditions])

        self.condition_names = dict(normal='Normal',
                                  diabetic='Reduced Metabolism',
                                  minute='Reduced Translation')

    @property
    def conditions(self):
        """ Environmental conditions. """
        return tuple(self.comparisons.keys())

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

            simulation (EnvironmentSimulation)

        """

        # load serialized simulation instance
        simulation = super(cls, cls).load(join(path, 'simulation.pkl'))

        # load simulation trajectories (if available)
        for condition, comparison in simulation.comparisons.items():

            # check that directory exists
            subdir = join(path, condition)
            if not isdir(subdir):
                continue

             # load simulation trajectories for control
            control_dir = join(subdir, 'control')
            if isdir(control_dir):
                comparison.reference = comparison.tstype.load(control_dir)

            # load simulation trajectories for perturbation
            perturbation_dir = join(subdir, 'perturbation')
            if isdir(perturbation_dir):
                comparison.compared = comparison.tstype.load(perturbation_dir)

        return simulation

    def save(self, path, saveall=False):
        """
        Save simulation to file. Simulations are saved as serialized pickle objects. TimeSeries data may optionally be saved as numpy arrays.

        Args:

            path (str) - save destination

            saveall (bool) - if True, save timeseries data

        """

        if saveall:
            for condition, comparison in self.comparisons.items():

                # make a directory
                subdir = join(path, condition)
                if not isdir(subdir):
                    mkdir(subdir)

                # save simulation trajectories
                comparison.reference.save(join(subdir, 'control'))
                comparison.compared.save(join(subdir, 'perturbation'))

        # save serialized object
        super().save(join(path, 'simulation.pkl'))

    def run(self, N=100, **kwargs):
        """
        Run simulation and evaluate comparison between wildtype and mutant for each environmental condition.

        Args:

            N (int) - number of independent simulation trajectories

            kwargs: keyword arguments for comparison

        """
        for condition in self.comparisons.keys():
            self.comparisons[condition] = super().run(condition, N=N, **kwargs)

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
