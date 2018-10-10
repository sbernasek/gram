import numpy as np
from scipy.integrate import trapz
import matplotlib.pyplot as plt
from genessa.timeseries import timeseries

# internal python imports
from ..figures.settings import *


class Comparison:

    """
    Base class for comparing a timeseries against a reference.

    Attributes:

        reference (TimeSeries) - reference timeseries

        compared (TimeSeries) - timeseries to be compared

        dim (int) - state space dimension to be compared

    """

    def __init__(self, reference, compared, dim=-1):
        """
        Instantiate timeseries comparison object.

        Args:

            reference (TimeSeries) - reference timeseries

            compared (TimeSeries) - timeseries to be compared

            dim (int) - state space dimension to be compared

        """

        # store attributes
        self.dim = dim

        # instantiate gaussian models for confidence band
        self.reference = timeseries.GaussianModel.from_timeseries(reference)
        self.compared = timeseries.GaussianModel.from_timeseries(compared)

    def plot_trajectories(self, colors=None, ax=None):
        """
        Plot simulated trajectories.

        Args:

            colors (tuple) - color for reference and compared trajectories

            ax (matplotlib.axes.AxesSubplot) - if None, create figure

        """

        # create figure if axes weren't provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(3, 2))

        if colors is None:
            colors = (('k',), ('r',))

        # plot trajectories
        self.reference.plot(dims=(self.dim,), ax=ax, colors=colors[0])
        self.compared.plot(dims=(self.dim,), ax=ax, colors=colors[1])


class AreaComparison(Comparison):
    """
    Class for comparing a timeseries against a reference. Comparison is based on evaluating the fraction of the compared timeseries' confidence band that does not intersect with the reference timeseries' confidence band.

    Attributes:

        bandwidth (float) - width of confidence band, 0 to 1

        below (float) - fraction of confidence band below the reference

        above (float) - fraction of confidence band above the reference

        error (float) - total non-overalpping fraction of confidence band

    Inherited Attributes:

        reference (TimeSeries) - reference timeseries

        compared (TimeSeries) - timeseries to be compared

        dim (int) - state space dimension to be compared

    """

    def __init__(self, reference, compared, dim=-1, bandwidth=.98):
        """
        Instantiate timeseries comparison object.

        Args:

            reference (TimeSeries) - reference timeseries

            compared (TimeSeries) - timeseries to be compared

            dim (int) - state space dimension to be compared

            bandwidth (float) - width of confidence band, 0 to 1

        """

        super().__init__(reference, compared, dim=dim)

        # store bandwidth
        self.bandwidth = bandwidth

        # evaluate comparison metric
        below, above = self.evaluate()
        self.below = below
        self.above = above
        self.error = below + above

    @staticmethod
    def integrate(t, y0, y1):
        """
        Integrate area between two 1D arrays.

        Args:

            t (np.ndarray[float]) - sample times

            y0 (np.ndarray[float]) - lower curve

            y1 (np.ndarray[float]) - upper curve

        Returns:

            area (float)

        """
        return trapz(y1-y0, t)

    def extract_bounds(self, model):
        """
        Extract lower and upper bounds for confidence band.

        Args:

            model (timeseries.GaussianModel)

        Returns:

            lower, upper (np.ndarray[float]) - bounds for confidence band

        """
        lower = model.norm.ppf((1-self.bandwidth)/2)[self.dim, :]
        upper = model.norm.ppf((1+self.bandwidth)/2)[self.dim, :]
        return lower, upper

    def extract_region_below(self, rbounds, cbounds):
        """
        Extract region that falls below reference confidence band.

        Args:

            rbounds (tuple) - lower and upper bounds for reference band

            cbounds (tuple) - lower and upper bounds for compared band

        Returns:

            t (np.ndarray[float]) - timepoints for region

            lbound (np.ndarray[float]) - lower bound for region

            ubound (np.ndarray[float]) - upper bound for region

        """
        ind = cbounds[0] < rbounds[0]
        lbound = cbounds[0][ind]
        ubound = np.vstack((rbounds[0][ind], cbounds[1][ind])).min(axis=0)

        return self.compared.t[ind], lbound, ubound

    def extract_region_above(self, rbounds, cbounds):
        """
        Extract region that falls above reference confidence band.

        Args:

            rbounds (tuple) - lower and upper bounds for reference band

            cbounds (tuple) - lower and upper bounds for compared band

        Returns:

            t (np.ndarray[float]) - timepoints for region

            lbound (np.ndarray[float]) - lower bound for region

            ubound (np.ndarray[float]) - upper bound for region

        """

        ind = cbounds[1] > rbounds[1]
        lbound = np.vstack((rbounds[1][ind], cbounds[0][ind])).max(axis=0)
        ubound = cbounds[1][ind]

        return self.compared.t[ind], lbound, ubound

    def evaluate(self):
        """
        Evaluate comparison.

        Returns:

            below (float) - fraction of confidence band below the reference

            above (float) - fraction of confidence band above the reference

        """

        # extract bounds for confidence bands
        rbounds = self.extract_bounds(self.reference)
        cbounds = self.extract_bounds(self.compared)

        # extract regions above and below reference
        t_b, lbound_b, ubound_b = self.extract_region_below(rbounds, cbounds)
        t_a, lbound_a, ubound_a = self.extract_region_above(rbounds, cbounds)

        # evalaute areas of non-overlapping regions of confidence band
        area_b = self.integrate(t_b, lbound_b, ubound_b)
        area_a = self.integrate(t_a, lbound_a, ubound_a)

        # evaluate total area of confidence band
        total_area = self.integrate(self.compared.t, cbounds[0], cbounds[1])

        # evaluate fraction of confidence band below and above reference
        below = area_b/total_area
        above = area_a/total_area

        return below, above

    def visualize(self, alpha=0.5, ax=None):
        """
        Visualize comparison.

        Args:

            alpha (float) - opacity for shaded regions of confidence band

            ax (matplotlib.axes.AxesSubplot) - if None, create figure

        """

        # create figure if axes weren't provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(3, 2))

        # extract bounds for confidence bands
        rbounds = self.extract_bounds(self.reference)
        cbounds = self.extract_bounds(self.compared)

        # plot confidence band for reference
        self.reference.plot_confidence_interval(ax=ax, alpha=0.2, colors='k')
        ax.plot(self.reference.t, rbounds[0], '-k')
        ax.plot(self.reference.t, rbounds[1], '-k')

        # plot confidence band for compared
        self.compared.plot_confidence_interval(ax=ax, alpha=0.2, colors='k')
        ax.plot(self.compared.t, cbounds[0], '--k')
        ax.plot(self.compared.t, cbounds[1], '--k')

        # shade regions above and below reference
        t_b, lbound_b, ubound_b = self.extract_region_below(rbounds, cbounds)
        t_a, lbound_a, ubound_a = self.extract_region_above(rbounds, cbounds)
        ax.fill_between(t_b, lbound_b, ubound_b, color='b', alpha=alpha)
        ax.fill_between(t_a, lbound_a, ubound_a, color='r', alpha=alpha)

        # format axis
        self.format_axis(ax)

    def format_axis(self, ax):
        """
        Format axis.

        Args:

            ax (matplotlib.axes.AxesSubplot)

        """

        ax.set_xlabel('Time (h)')

        # display comparison metrics
        self.display_metrics(ax)

    def display_metrics(self, ax):
        """
        Display comparison metrics on axes.

        Args:

            ax (matplotlib.axes.AxesSubplot)

        """

        x = ax.get_xlim()[1] - 0.05*ax.get_xticks().ptp()
        y = ax.get_ylim()[1] - 0.05*ax.get_yticks().ptp()

        kw = dict(ha='right', va='top')
        ax.text(x, y, '{:0.1%} error'.format(self.error), **kw)
        ax.text(x, y, '\n{:0.1%} above'.format(self.above), color='r', **kw)
        ax.text(x, y, '\n\n{:0.1%} below'.format(self.below), color='b', **kw)
