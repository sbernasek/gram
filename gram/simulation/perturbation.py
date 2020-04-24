# external package imports

# internal imports
from ..analysis.comparison import Comparison, PromoterComparison, GaussianComparison, MultiComparison
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

    def simulate(self, condition='normal', N=5000, seed=None, debug=False):
        """
        Run simulation under the specified conditions.

        Args:

            condition (str) - simulation conditions affecting rate parameters

            N (int) - number of independent simulation trajectories

            seed (int) - seed for random number generator

            debug (bool) - if True, use debugging mode

        Returns:

            before (genessa TimeSeries) - trajectories before perturbation

            after (genessa TimeSeries) - trajectories after perturbation

        """

        # instantiate input signal
        signal = self.build_signal(condition)

        # run simulations
        kwargs = dict(N=N, seed=seed, debug=debug)
        before = super().simulate(self.cell, signal, condition, **kwargs)
        after = super().simulate(self.mutant, signal, condition, **kwargs)

        return before, after

    def compare(self,
                reference,
                compared,
                mode=None,
                horizon=0,
                deviations=False,
                **kwargs):
        """
        Compare simulation trajectories between two conditions.

        Args:

            reference (genessa TimeSeries) - reference trajectories

            compared (genessa TimeSeries) - compared trajectories

            mode (str) - comparison type, options are:
                empirical (default): fraction of trajectories outside reference confidence band
                gaussian: fraction of gaussian model below/above reference confidence band
                promoters: fraction of trajectories below reference at peak of expression

            horizon (float) - duration of comparison, 0 if unlimited

            deviations (bool) - if True, compare deviations from initial value

            kwargs: keyword arguments for comparison

        Returns:

            comparison (Comparison derivative)

        """

        # convert timeseires to deviation values
        if deviations:
            reference = reference.get_deviations(values='final')
            compared = compared.get_deviations(values='final')

        # crop timeseries
        start = self.pulse_start  / self.timescale
        if horizon is None or horizon == 0:
            stop = self.simulation_duration / self.timescale
        else:
            stop = start + horizon
        reference = reference.crop(start, stop)
        compared = compared.crop(start, stop)

        # evaluate comparison
        if mode == 'promoters':
            comparison = PromoterComparison(reference, compared, **kwargs)
        elif mode == 'gaussian':
            comparison = GaussianComparison(reference, compared, **kwargs)
        elif mode == 'multi':
            comparison = MultiComparison(reference, compared, **kwargs)
        else:
            comparison = Comparison(reference, compared, **kwargs)

        return comparison
