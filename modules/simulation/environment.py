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

    def run(self, N=100, **kwargs):
        """
        Run simulation and evaluate comparison between wildtype and mutant for each environmental condition.

        Args:

            N (int) - number of independent simulation trajectories

            kwargs: keyword arguments for comparison

        """
        for condition in self.comparisons.keys():
            self.comparisons[condition] = super().run(condition, N=N, **kwargs)

    def visualize(self, axes=None):
        """
        Visualize comparison for each environmental condition.

        Args:

            axes (tuple) - matplotlib.axes.AxesSubplot for each condition

        """

        # create axes if none were provided
        if axes is None:
            ncols = self.N
            figsize=(ncols*3, 2)
            fig, axes = plt.subplots(1, ncols, sharey=True, figsize=figsize)

        # visualize comparison under each condition
        for i, (condition, comparison) in enumerate(self.comparisons.items()):
            comparison.visualize(ax=axes[i])
            axes[i].set_title(self.condition_names[condition])

        axes[0].set_ylabel('Protein level')
