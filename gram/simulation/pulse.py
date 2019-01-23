# external package imports
import numpy as np
import warnings
from copy import copy
import pickle
from genessa.signals.signals import cSignal, cSquarePulse
from genessa.solver.deterministic import DeterministicSimulation
from genessa.solver.stochastic import MonteCarloSimulation
from genessa.solver.debug import Debugger

# internal imports
from .parameters import signal_sensitivity


class PulseSimulation:
    """
    Numerical simulation of a single gene expression pulse.

    Attributes:

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

    def __init__(self, cell,
                 pulse_start=100,
                 pulse_duration=3,
                 pulse_baseline=0,
                 pulse_magnitude=1,
                 pulse_sensitive=False,
                 simulation_duration=500,
                 dt=1,
                 timescale=60):
        """
        Instantiate pulse simulation.

        Args:

            cell (Cell derivative)

            pulse_start (float) - pulse onset time

            pulse_duration (float) - pulse duration under normal conditions

            pulse_baseline (float) - basal signal level

            pulse_magnitude (float) - magnitude of pulse

            pulse_sensitive (bool) - if True, pulse duration depends upon environmental conditions

            simulation_duration (float) - simulation duration

            dt (float) - sampling interval

            timescale (float) - time scaling factor

        """

        self.cell = cell
        self.timescale = timescale
        self.simulation_duration = simulation_duration * timescale
        self.dt = dt * timescale
        self.pulse_start = pulse_start * timescale
        self.pulse_duration = pulse_duration * timescale
        self.pulse_baseline = pulse_baseline
        self.pulse_magnitude = pulse_magnitude
        self.pulse_sensitive = pulse_sensitive

    @staticmethod
    def load(path):
        """
        Load simulation from file.

        Args:

            path (str) - file path

        Returns:

            simulation (PulseSimulation derivative)

        """
        with open(path, 'rb') as file:
            simulation = pickle.load(file)
        return simulation

    def save(self, path):
        """
        Save simulation to file. Simulations are saved as serialized pickle objects.

        Args:

            path (str) - filepath

        """
        with open(path, 'wb') as file:
            pickle.dump(self, file, protocol=-1)

    def evaluate_steady_state(self, cell, condition='normal', tol=1e-2):
        """
        Returns steady state levels under specified condition.

        Args:

            cell (Cell instance)

            condition (str) - simulation conditions affecting rate parameters

            tol (float) - maximum percent change across last five timepoints

        Returns:

            ss (np.ndarray) - steady state levels

        """

        # define constant basal-level signal
        signal = cSignal(self.pulse_baseline)

        # run deterministic simulation to determine steady state
        simulation = DeterministicSimulation(cell, condition=condition)

        # run ODE solver (ignoring divide by zero warning)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            results = simulation.solve_ivp(
                signal=signal,
                duration=2*self.simulation_duration)

        # take last timepoint as steady state
        ss = results.mean[:, -1]

        # if all states are zero at steady state, return steady state
        if np.all(ss==0):
            return ss

        # if solver exploded, return zeros and file a warning
        elif np.abs(ss).max() > 1e10:
            warnings.warn('Solver did not converge.', UserWarning)
            return np.zeros(ss.size, dtype=np.float64)

        # if solver returned negative values, return zeros and file a warning
        elif ss.min() < 0:
            warnings.warn('Steady state is negative.', UserWarning)
            return np.zeros(ss.size, dtype=np.float64)

        # otherwise ensure that maximum change is below specified tolerance
        else:
            x_ss = ss[ss>0].reshape(-1, 1)
            dx = np.diff(results.mean[ss>0, -5:], axis=1)
            if np.abs(dx/x_ss).max() >= tol:
                warnings.warn('Did not reach steady state.', UserWarning)
            return ss

    def build_signal(self, condition):
        """
        Returns pulsatile signal object.

        Args:

            condition (str) - simulation conditions affecting rate parameters

        Returns:

            signal (genessa cSquarePulse)

        """

        # get pulse duration and magnitude for regular conditions
        pulse_duration = copy(self.pulse_duration)
        pulse_magnitude = copy(self.pulse_magnitude)

        # scale pulse duration and magnitude with environmental conditions
        if self.pulse_sensitive:
            pulse_duration *= signal_sensitivity[condition]['duration']
            pulse_magnitude *= signal_sensitivity[condition]['magnitude']

        # instantiate signal
        signal = cSquarePulse(t_on=self.pulse_start,
                              t_off=self.pulse_start+pulse_duration,
                              off=self.pulse_baseline,
                              on=self.pulse_baseline+pulse_magnitude)

        return signal

    def simulate(self,
                 cell,
                 signal,
                 condition='normal',
                 N=5000,
                 seed=None,
                 debug=False):
        """
        Run simulation under the specified conditions for a specified cell.

        Args:

            cell (Cell instance)

            signal (genessa.cSignalType) - input signal

            condition (str) - simulation conditions affecting rate parameters

            N (int) - number of independent simulation trajectories

            seed (int) - seed for random number generator

            debug (bool) - if True, use debugging mode

        Returns:

            timeseries (genessa TimeSeries)

        """

        # use steady states as initial condition
        ic = self.evaluate_steady_state(cell, condition=condition)

        # instantiate stochastic solver
        if debug:
            print('RUNNING SIMULATION IN DEBUG MODE.')
            sim = Debugger(cell, condition, ic=ic, seed=seed)
        else:
            sim = MonteCarloSimulation(cell, condition, ic=ic, seed=seed)

        # run stochastic simulation
        ts = sim.run(N=N,
                     signal=signal,
                     duration=self.simulation_duration,
                     dt=self.dt)

        # apply timescale
        ts.t /= self.timescale

        return ts

    def run(self,
            condition='normal',
            N=100,
            seed=None,
            debug=False):
        """
        Run simulation under the specified conditions.

        Args:

            condition (str) - simulation conditions affecting rate parameters

            N (int) - number of independent simulation trajectories

            seed (int) - seed for random number generator

            debug (bool) - if True, use debugging mode

        Returns:

            timeseries (genessa TimeSeries)

        """

        # instantiate input signal
        signal = self.build_signal(condition)

        ts = self.simulate(
            self.cell,
            signal,
            condition=condition,
            N=N,
            seed=seed,
            debug=debug)

        return ts
