# external package imports

# internal imports
from ..analysis.comparison import Comparison, AreaComparison, CDFComparison
from ..analysis.comparison import ThresholdComparison
from .pulse import PulseSimulation


class PerturbationSimulation(PulseSimulation):
    """
    Numerical simulations of a single gene expression pulse before and after a genetic perturbations.

    Attributes:

        mutant (Cell derivative) - cell with perturbation applied

    Inherited Attributes:

        cell (Cell derivative)

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
        Instantiate pulse simulation.

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

        # apply perturbation to generate mutant
        self.mutant = self.cell.perturb()

    def simulate(self, condition='normal', N=100):
        """
        Run simulation under the specified conditions.

        Args:

            condition (str) - simulation conditions affecting rate parameters

            N (int) - number of independent simulation trajectories

        Returns:

            before (genessa TimeSeries) - trajectories before perturbation

            after (genessa TimeSeries) - trajectories after perturbation

        """

        # instantiate input signal
        signal = self.build_signal(condition)

        # run simulations
        before = super().simulate(self.cell, signal, condition, N=N)
        after = super().simulate(self.mutant, signal, condition, N=N)

        return before, after

    @staticmethod
    def compare(reference, compared, mode=None, deviations=False, **kwargs):
        """
        Compare simulation trajectories between two conditions.

        Args:

            reference (genessa TimeSeries) - reference trajectories

            compared (genessa TimeSeries) - compared trajectories

            mode (str) - comparison type, options are:
                empirical: fraction of trajectories below/above reference
                area: fraction of confidence band area below/above reference
                cdf: fraction of gaussian model below/above reference
                threshold: fraction of gaussian model above threshold

            deviations (bool) - if True, compare deviations from initial value

            kwargs: keyword arguments for comparison

        Returns:

            comparison (Comparison derivative)

        """

        if mode == 'empirical' or mode is None:
            comparison = Comparison(reference, compared, deviations, **kwargs)
        elif mode == 'area':
            comparison = AreaComparison(reference, compared, deviations, **kwargs)
        elif mode == 'cdf':
            comparison = CDFComparison(reference, compared, deviations, **kwargs)
        elif mode == 'threshold':
            comparison = ThresholdComparison(reference, compared, deviations, **kwargs)

        return comparison
