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

            ts_wildtype (genessa TimeSeries) - timeseries before perturbation

            ts_mutant (genessa TimeSeries) - timeseries after perturbation

        """

        # instantiate input signal
        signal = self.build_signal(condition)

        # run simulations
        ts_wildtype = super().simulate(self.cell, signal, condition, N=N)
        ts_mutant = super().simulate(self.mutant, signal, condition, N=N)

        return ts_wildtype, ts_mutant

    def run(self, condition='normal', N=100, comparison_type=None, **kwargs):
        """
        Run simulation under the specified conditions and compare dynamics between wildtype and mutant.

        Args:

            condition (str) - simulation conditions affecting rate parameters

            N (int) - number of independent simulation trajectories

            comparison_type (str) - comparison type e.g. 'empirical', 'area', 'cdf' or 'threshold'. Defaults to 'empirical'.

            kwargs: keyword arguments for comparison

        Returns:

            comparison (Comparison derivative)

        """

        # run simulation
        ts0, ts1 = self.simulate(condition, N)

        # evaluate comparison
        if comparison_type == 'empirical' or comparison_type is None:
            comparison = Comparison(ts0, ts1, **kwargs)
        elif comparison_type == 'area':
            comparison = AreaComparison(ts0, ts1, **kwargs)
        elif comparison_type == 'cdf':
            comparison = CDFComparison(ts0, ts1, **kwargs)
        elif comparison_type == 'threshold':
            comparison = ThresholdComparison(ts0, ts1, **kwargs)

        return comparison
