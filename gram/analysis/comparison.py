import numpy as np
from scipy.integrate import trapz
import matplotlib.pyplot as plt
from genessa.timeseries.gaussian import GaussianModel
from matplotlib.collections import LineCollection

# internal python imports
from ..figures.settings import *


class ComparisonMethods:
    """ Methods for comparison objects. """

    @staticmethod
    def integrate(t, y, indices=None):
        """
        Integrate 1D array.

        Args:

            t (np.ndarray[float]) - sample times

            y (np.ndarray[float]) - sample values

            indices (list) - list of indices of contiguous subsegments

        Returns:

            area (float)

        """
        if indices is None:
            return trapz(y, t)
        else:
            return sum([trapz(y[ind], t[ind]) for ind in indices])

    @staticmethod
    def extract_region_below(rbounds, cbounds):
        """
        Extract region that falls below reference confidence band.

        Args:

            rbounds (tuple) - lower and upper bounds for reference band

            cbounds (tuple) - lower and upper bounds for compared band

        Returns:

            indices (list) - list of indices of segments below reference

            lbound (np.ndarray[float]) - lower bound for region

            ubound (np.ndarray[float]) - upper bound for region

        """
        below = cbounds[0] < rbounds[0]
        indices = np.split(np.arange(below.size),1+np.diff(below).nonzero()[0])
        indices = [ind for ind in indices if np.all(below[ind])]
        lbound = cbounds[0]
        ubound = np.vstack((rbounds[0], cbounds[1])).min(axis=0)
        return indices, lbound, ubound

    @staticmethod
    def extract_region_above(rbounds, cbounds):
        """
        Extract region that falls above reference confidence band.

        Args:

            rbounds (tuple) - lower and upper bounds for reference band

            cbounds (tuple) - lower and upper bounds for compared band

        Returns:

            indices (list) - list of indices of segments above reference

            lbound (np.ndarray[float]) - lower bound for region

            ubound (np.ndarray[float]) - upper bound for region

        """
        above = cbounds[1] > rbounds[1]
        indices = np.split(np.arange(above.size),1+np.diff(above).nonzero()[0])
        indices = [ind for ind in indices if np.all(above[ind])]
        lbound = np.vstack((rbounds[1], cbounds[0])).max(axis=0)
        ubound = cbounds[1]
        return indices, lbound, ubound


class ComparisonProperties:
    """

    Properties for comparison methods.

    Properties:

        t (np.ndarray[float]) - reference timepoints

        _peak_index (int) - time index of peak expression

        _peak_time (float) - time of peak expression

        _comparison_index (int) - time index of comparison

        _comparison_time (float) - time of comparison

        lower (np.ndarray[float]) - lower bound for reference trajectories

        upper (np.ndarray[float]) - upper bound for reference trajectories

        fractions_below (np.ndarray[float]) - fractions below lower bound

        fractions_above (np.ndarray[float]) - fractions above upper bound

    """

    @property
    def t(self):
        """ Reference timepoints. """
        return self.reference.t

    # @property
    # def threshold(self):
    #     """ Commitment threshold. """
    #     return self.reference.peaks[self.dim] * self.fraction_of_max

    @property
    def threshold(self):
        """ Commitment threshold. """
        return self.upper[self._comparison_index]

    @property
    def _peak_index(self):
        """ Index of peak expression. """
        return self.reference.peak_indices[self.dim]

    @property
    def _peak_time(self):
        """ Time of peak expression. """
        return self.t[self._peak_index]

    @property
    def _comparison_index(self):
        """ Index of time at which reference reaches threshold. """

        # evaluate population mean at comparison time
        final_mean = self.reference.peaks[self.dim] * self.fraction_of_max
        peak_index = self.reference.peak_indices[self.dim]

        # determine first time at which mean reaches final level
        indices = self.reference.index(final_mean, self.dim, mode='mean')
        if indices.size == 0 or indices[-1] <= peak_index:
            return None
        else:
            return indices[-1]

    # @property
    # def _comparison_index(self):
    #     """ Index of time at which reference reaches threshold. """
    #     indices = self.reference.index(self.threshold, self.dim, mode='upper')

    #     if indices.size == 0 or indices[-1] == 0:
    #         return None
    #     else:
    #         return indices[-1]

    @property
    def _comparison_time(self):
        """ Time at which reference reaches threshold. """
        return self.t[self.comparison_index]

    @property
    def lower(self):
        """ Lower bound of reference. """
        q = (100-self.bandwidth)/2
        return self.reference.evaluate_quantile(q)[self.dim]

    @property
    def upper(self):
        """ Upper bound of reference. """
        q = (100+self.bandwidth)/2
        return self.reference.evaluate_quantile(q)[self.dim]

    @property
    def fractions_below(self):
        """ Fractions of trajectories below the lowest reference. """
        return (self.compared.states[:, self.dim, :] < self.lower).mean(axis=0)

    @property
    def fractions_above(self):
        """ Fractions of trajectories above the highest reference. """
        return (self.compared.states[:, self.dim, :] > self.upper).mean(axis=0)


class ComparisonVis:
    """
    Visualization methods for comparison objects.
    """

    def shade_outlying_areas(self,
                             alpha=0.2,
                             reference_color='k',
                             compared_color='k',
                             above_color='r',
                             above_alpha=0.5,
                             below_color='b',
                             below_alpha=0.5,
                             ax=None,
                             show_threshold=False):
        """
        Visualize comparison by shading the region encompassing trajectories that lie below or above all reference trajectories.

        Args:

            alpha (float) - opacity for shaded regions of confidence band

            reference_color (str) - color for reference confidence band

            compared_color (str) - color for compared confidence band

            above_color, below_color (str) - colors for above/below reference

            ax (matplotlib.axes.AxesSubplot) - if None, create figure

            show_threshold (bool) - if True, show threshold definition

        """

        # create figure if axes weren't provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(3, 2))

        # extract bounds for confidence bands
        tf = self.comparison_index + 1

        t = self.t[:tf]
        rbounds = (self.lower[:tf], self.upper[:tf])
        cbounds = (self.compared.lower[self.dim][:tf],
                   self.compared.upper[self.dim][:tf])

        # plot confidence band for reference
        ax.fill_between(t, *rbounds, color=reference_color, alpha=alpha)
        ax.plot(t, rbounds[0], '-k')
        ax.plot(t, rbounds[1], '-k')

        # plot confidence band for compared
        ax.fill_between(t, *cbounds, color=compared_color, alpha=alpha)
        ax.plot(t, cbounds[0], '--k')
        ax.plot(t, cbounds[1], '--k')

        # shade regions below reference
        ind_b, lbound_b, ubound_b = self.extract_region_below(rbounds, cbounds)
        for ind in ind_b:
            ax.fill_between(t[ind],
                            lbound_b[ind],
                            ubound_b[ind],
                            color=below_color,
                            alpha=below_alpha)

        # shade regions above reference
        ind_a, lbound_a, ubound_a = self.extract_region_above(rbounds, cbounds)
        for ind in ind_a:
            ax.fill_between(t[ind],
                            lbound_a[ind],
                            ubound_a[ind],
                            color=above_color,
                            alpha=above_alpha)

        # display threshold definition
        if show_threshold:
            self.display_threshold_definition(ax)

        # format axis
        self.format_axis(ax)

    def plot_outlying_trajectories(self, ax=None, show_threshold=False):
        """
        Visualize comparison by plotting the trajectories that lie below or above the reference trajectories.

        Args:

            ax (matplotlib.axes.AxesSubplot) - if None, create figure

            show_threshold (bool) - if True, show threshold definition

        """

        # create figure if axes weren't provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(3, 2))

        # extract bounds for confidence bands
        tf = self.comparison_index + 1
        lower, upper = self.lower[:tf], self.upper[:tf]
        t = self.t[:tf]

        # plot confidence band for reference
        ax.fill_between(t, lower, upper, color='k', alpha=0.2)
        ax.plot(t, lower, '-k')
        ax.plot(t, upper, '-k')

        # assemble segments of trajectories below/above reference extrema
        segments_below, segments_within, segments_above = [], [], []
        for x in self.compared.states[:, self.dim, :tf]:
            below, above = x<lower, x>upper

            # select outlying line segments
            ind_b = np.split(np.arange(x.size), np.diff(below).nonzero()[0]+1)
            ib = list(filter(lambda i: np.all(below[i]), ind_b))
            ind_a = np.split(np.arange(x.size), np.diff(above).nonzero()[0]+1)
            ia = list(filter(lambda i: np.all(above[i]), ind_a))
            iw = list(filter(lambda i: not np.all(above[i]), ind_a))

            # append line segments to lists
            segments_below.extend([list(zip(t[i], x[i])) for i in ib])
            segments_above.extend([list(zip(t[i], x[i])) for i in ia])
            segments_within.extend([list(zip(t[i], x[i])) for i in iw])

        # compile line objects
        lines_below = LineCollection(segments_below, colors='b')
        lines_above = LineCollection(segments_above, colors='r')
        lines_within = LineCollection(segments_within, colors='k', alpha=0.1)

        # add lines to plot
        for lines in (lines_below, lines_within, lines_above):
            ax.add_collection(lines)

        # display threshold definition
        if show_threshold:
            self.display_threshold_definition(ax)

        # format axis
        self.format_axis(ax)

    def display_threshold_definition(self, ax):
        """
        Display arrows defining threshold and commitment time.

        Note: will likely crash if axes limits aren't set

        Args:

            ax (matplotlib.axes.AxesSubplot)

        """

        # plot threshold geometry
        peak_time = self._peak_time
        comparison_time = self.comparison_time
        peak_value = self.reference.peaks[self.dim]
        max_error = self.compared.upper[self.dim][self.comparison_index]

        # add vertical arrow defining threshold value
        ax.annotate(s='',
                    xy=(peak_time, self.threshold),
                    xytext=(peak_time, peak_value),
                    arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0))

        # add horizontal arrow defining commitment time
        ax.annotate(s='',
                    xy=(peak_time, self.threshold),
                    xytext=(comparison_time, self.threshold),
                    arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0))

        # add vertical arrow defining error
        ax.annotate(s='',
                    xy=(1+comparison_time, self.threshold),
                    xytext=(1+comparison_time, max_error),
                    arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0, color='k'))

        # annotate error
        ax.text(comparison_time+1.5,
                (self.threshold+max_error)/2,
                'error',
                ha='left',
                va='center')

    def format_axis(self, ax):
        """
        Format axis.

        Args:

            ax (matplotlib.axes.AxesSubplot)

        """

        ax.set_xlabel('Time (h)')

        # display comparison metrics
        #self.display_metrics(ax)

    def display_metrics(self, ax, **kwargs):
        """
        Display comparison metrics on axes.

        Args:

            ax (matplotlib.axes.AxesSubplot)

        """

        x = ax.get_xlim()[1] - 0.05*ax.get_xticks().ptp()
        y = ax.get_ylim()[1] - 0.05*ax.get_yticks().ptp()

        kw = dict(ha='right', va='top', fontsize=8)
        ax.text(x, y, '{:0.1%} error'.format(self.error), **kw)
        ax.text(x, y, '\n{:0.1%} above'.format(self.above), color='r', **kw)
        ax.text(x, y, '\n\n{:0.1%} below'.format(self.below), color='b', **kw)


class Comparison(ComparisonProperties, ComparisonMethods, ComparisonVis):

    """
    Base class for comparing a timeseries against a reference.

    Comparison is based on evaluating the fraction of trajectories that lie above or below the reference trajectory confidence band.

    Attributes:

        reference (TimeSeries) - reference timeseries

        compared (TimeSeries) - timeseries to be compared

        bandwidth (float) - width of confidence band

        fraction_of_max (float) - fraction of peak mean reference value used to define commitment time

        dim (int) - state space dimension to be compared

        below (float) - fraction of confidence band below the reference

        above (float) - fraction of confidence band above the reference

        error (float) - total non-overlapping fraction of confidence band

        below_threshold (float) - fraction below lower threshold

        above_threshold (float) - fraction above upper threshold

        threshold_error (float) - fraction outside thresholds

        reached_comparison (bool) - if True, simulation reached comparison time

        tstype (type) - python class for timeseries objects

    Properties:

        t (np.ndarray[float]) - reference timepoints

        _peak_index (int) - time index of peak expression

        _peak_time (float) - time of peak expression

        _comparison_index (int) - time index of comparison

        _comparison_time (float) - time of comparison

        lower (np.ndarray[float]) - lower bound for reference trajectories

        upper (np.ndarray[float]) - upper bound for reference trajectories

        fractions_below (np.ndarray[float]) - fractions below lower bound

        fractions_above (np.ndarray[float]) - fractions above upper bound

    """

    def __init__(self, reference, compared,
                 bandwidth=98,
                 fraction_of_max=0.3,
                 dim=-1):
        """
        Instantiate timeseries comparison object.

        Args:

            reference (TimeSeries) - reference timeseries

            compared (TimeSeries) - timeseries to be compared

            bandwidth (float) - width of confidence band, 0 to 100

            fraction_of_max (float) - fraction of peak mean reference value used to define commitment time

            dim (int) - state space dimension to be compared

        """

        # store simulation trajectories
        self.reference = reference
        self.compared = compared

        # store attributes
        self.bandwidth = bandwidth
        self.fraction_of_max = fraction_of_max
        self.dim = dim
        self.tstype = self.reference.__class__

        # evaluate comparison index and time
        self.compare()

    def __getstate__(self):
        """ Returns all attributes except TimeSeries instances. """
        excluded = ('reference', 'compared')
        return {k: v for k, v in self.__dict__.items() if k not in excluded}

    def compare(self):
        """ Run comparison procedure. """

        # determine whether commitment threshold is reached
        self.comparison_index = self._comparison_index
        if self.comparison_index is None:
            self.reached_comparison = False
        else:
            self.reached_comparison = True

        # evaluate comparison metric
        if self.reached_comparison:
            self.comparison_time = self.t[self.comparison_index]

            # evaluate integrated error
            below, above = self.evaluate()
            self.below = below
            self.above = above
            self.error = below + above

            # evaluate threshold error
            below_threshold, above_threshold = self.evaluate_threshold()
            self.below_threshold = below_threshold
            self.above_threshold = above_threshold
            self.threshold_error = below_threshold + above_threshold

    def evaluate(self):
        """
        Evaluate comparison.

        Returns:

            below (float) - mean fraction of trajectories below the reference

            above (float) - mean fraction of trajectories above the reference

        """

        # determine start index (pulse onset)
        #ind = self.reference.mean[self.dim].nonzero()[0][0] + 1

        t0 = 0
        tf = self.comparison_index

        # evalaute fractions below/above confidence band
        t = self.t[t0: tf] - self.t[t0]
        t_normalized = t / t.max()

        # determine correction factor
        correction = (100-self.bandwidth)/2/100

        # fraction below
        fractions_below = self.fractions_below[t0: tf]
        fractions_below -= correction
        fractions_below[fractions_below<0] = 0

        # fraction above
        fractions_above = self.fractions_above[t0: tf]
        fractions_above -= correction
        fractions_above[fractions_above<0] = 0

        below = self.integrate(t_normalized, fractions_below)
        above = self.integrate(t_normalized, fractions_above)

        return below, above

    def evaluate_threshold(self):
        """
        Evaluate comparison.

        Returns:

            below (float) - mean fraction of trajectories below the reference

            above (float) - mean fraction of trajectories above the reference

        """

        below = self.fractions_below[self.comparison_index]
        above = self.fractions_above[self.comparison_index]

        # apply correction
        correction = (100-self.bandwidth)/2/100
        below -= correction
        above -= correction

        return max(below, 0), max(above, 0)


class GaussianComparison(Comparison):
    """
    Class for comparing a timeseries against a reference. Comparison is based on evaluating the fraction of the compared timeseries that lies above or below the reference timeseries.

    Attributes:

        reference (GaussianModel) - reference timeseries

        compared (GaussianModel) - timeseries to be compared

        tskwargs (dict) - keyword arguments for timeseries instantiation

    Inherited Attributes:

        dim (int) - state space dimension to be compared

        below (float) - fraction of confidence band below the reference

        above (float) - fraction of confidence band above the reference

        error (float) - total non-overalpping fraction of confidence band

        tstype (type) - python class for timeseries objects

    Properties:

        t (np.ndarray[float]) - reference timepoints

        lower (np.ndarray[float]) - lower bound for reference trajectories

        upper (np.ndarray[float]) - upper bound for reference trajectories

        fractions_below (np.ndarray[float]) - fractions below lower bound

        fractions_above (np.ndarray[float]) - fractions above upper bound

    """

    def __init__(self,
                 reference,
                 compared,
                 bandwidth=98,
                 dim=-1):
        """
        Instantiate timeseries comparison object.

        Args:

            reference (TimeSeries) - reference timeseries

            compared (TimeSeries) - timeseries to be compared

            bandwidth (float) - width of confidence band, 0 to 100

            dim (int) - state space dimension to be compared

        """

        # fit gaussian models to timeseries
        reference = GaussianModel.from_timeseries(reference, bandwidth/100)
        compared = GaussianModel.from_timeseries(compared, bandwidth/100)

        # call parent instantiation (runs evaluation)
        super().__init__(reference, compared, bandwidth=bandwidth, dim=dim)

        # store timeseries kwargs
        self.tskwargs = dict(bandwidth=bandwidth)

    # @property
    # def lower(self):
    #     """ Lower bound of reference. """
    #     return self.reference.lower[self.dim]

    # @property
    # def upper(self):
    #     """ Upper bound of reference. """
    #     return self.reference.upper[self.dim]

    @property
    def lower(self):
        """ Lower bound of reference. """
        q = (100-self.bandwidth)/2/100
        return self.reference.norm.ppf(q)[self.dim]

    @property
    def upper(self):
        """ Upper bound of reference. """
        q = (100+self.bandwidth)/2/100
        return self.reference.norm.ppf(q)[self.dim]

    @property
    def fractions_below(self):
        """ Fractions of trajectories below the lowest reference. """
        return self.compared.norm.cdf(self.lower)[self.dim]

    @property
    def fractions_above(self):
        """ Fractions of trajectories above the highest reference. """
        return 1 - self.compared.norm.cdf(self.upper)[self.dim]


class PromoterComparison(Comparison):
    """
    Comparison method for promoter perturbations.

    Uses commitment time based on lower bound, and evaluates threshold at time of peak expression.
    """

    @property
    def _comparison_index(self):
        """ Index of time at which reference reaches threshold. """
        indices = self.reference.index(self.threshold, self.dim, mode='lower')

        if indices.size == 0 or indices[-1] == 0:
            return None
        else:
            return indices[-1]

    def evaluate_threshold(self):
        """
        Evaluate threshold comparison.

        Returns:

            below (float) - mean fraction of trajectories below the reference

            above (float) - mean fraction of trajectories above the reference

        """
        below = self.fractions_below[self._peak_index]
        above = self.fractions_above[self._peak_index]
        return below, above


class MultiComparison(Comparison):

    """
    Class for making multiple comparisons of a timeseries against a reference.

    Comparison is based on evaluating the fraction of trajectories that lie above or below the reference trajectory confidence band. Comparison is evaluated at multiple timepoints.

    Attributes:

        reference (TimeSeries) - reference timeseries

        compared (TimeSeries) - timeseries to be compared

        bandwidth (float) - width of confidence band

        fraction_of_max (np.ndarray[float]) - fractions of peak mean reference value used to define commitment time

        dim (int) - state space dimension to be compared

        below (np.ndarray[float]) - fractions of confidence band below the reference

        above (np.ndarray[float]) - fractions of confidence band above the reference

        error (np.ndarray[float]) - total non-overlapping fraction of confidence band

        below_threshold (np.ndarray[float]) - fraction below lower threshold

        above_threshold (np.ndarray[float]) - fraction above upper threshold

        threshold_error (np.ndarray[float]) - fraction outside thresholds

        reached_comparison (np.ndarray[bool]) - if True, simulation reached comparison time

        tstype (type) - python class for timeseries objects

    Properties:

        t (np.ndarray[float]) - reference timepoints

        _peak_index (int) - time index of peak expression

        _peak_time (float) - time of peak expression

        _comparison_index (np.ndarray[int]) - time index of comparison

        _comparison_time (np.ndarray[float]) - time of comparison

        lower (np.ndarray[float]) - lower bound for reference trajectories

        upper (np.ndarray[float]) - upper bound for reference trajectories

        fractions_below (np.ndarray[float]) - fractions below lower bound

        fractions_above (np.ndarray[float]) - fractions above upper bound

    """

    def __init__(self, reference, compared,
                 bandwidth=98,
                 dim=-1):
        """
        Instantiate timeseries multiple-comparison object.

        Args:

            reference (TimeSeries) - reference timeseries

            compared (TimeSeries) - timeseries to be compared

            bandwidth (float) - width of confidence band, 0 to 100

            dim (int) - state space dimension to be compared

        """

        # store simulation trajectories
        self.reference = reference
        self.compared = compared

        # store attributes
        self.fraction_of_max = np.arange(0.1, 1., .1)[::-1]
        self.bandwidth = bandwidth
        self.dim = dim
        self.tstype = self.reference.__class__

        # evaluate comparison index and time
        self.compare()

    @property
    def _comparison_index(self):
        """ Index of time at which reference reaches threshold. """

        # evaluate population means at comparison times
        final_means = self.reference.peaks[self.dim] * self.fraction_of_max
        peak_index = self.reference.peak_indices[self.dim]

        def find_value(value):
            """ Determine first time at which mean reaches <value>. """
            indices = self.reference.index(value, self.dim, mode='mean')
            if indices.size == 0 or indices[-1] <= peak_index:
                return None
            else:
                return indices[-1]

        return np.array([find_value(x) for x in final_means])

    @property
    def _comparison_time(self):
        """ Time at which reference reaches threshold. """
        return self.t[self.comparison_index]

    def compare(self):
        """ Run comparison procedure. """

        # determine whether commitment threshold is reached
        self.comparison_index = self._comparison_index
        self.reached_comparison = (self.comparison_index != None)

        comparison_time = []
        below, above, error = [], [], []
        below_threshold, above_threshold, threshold_error = [], [], []

        for i, comparison_index in enumerate(self.comparison_index):

            # evaluate comparison metric
            if not self.reached_comparison[i]:
                comparison_time.append(None)
                below.append(None)
                above.append(None)
                error.append(None)
                below_threshold.append(None)
                above_threshold.append(None)
                threshold_error.append(None)

            else:
                comparison_time.append(self.t[comparison_index])

                # evaluate integrated error
                b, a = self.evaluate(comparison_index, t0=0)
                below.append(b)
                above.append(a)
                error.append(b+a)

                # evaluate threshold error
                bt, at = self.evaluate_threshold(comparison_index)
                below_threshold.append(bt)
                above_threshold.append(at)
                threshold_error.append(bt + at)

        self.comparison_time = np.array(comparison_time)
        self.below = np.array(below)
        self.above = np.array(above)
        self.error = np.array(error)
        self.below_threshold = np.array(below_threshold)
        self.above_threshold = np.array(above_threshold)
        self.threshold_error = np.array(threshold_error)

    def evaluate(self, tf, t0=0):
        """
        Evaluate comparison.

        Args:

            tf (int) - time index for comparison

            t0 (int) - time index for integration start

        Returns:

            below (float) - mean fraction of trajectories below the reference

            above (float) - mean fraction of trajectories above the reference

        """

        # evalaute fractions below/above confidence band
        t = self.t[t0: tf] - self.t[t0]
        t_normalized = t / t.max()

        # determine correction factor
        correction = (100-self.bandwidth)/2/100

        # fraction below
        fractions_below = self.fractions_below[t0: tf]
        fractions_below -= correction
        fractions_below[fractions_below<0] = 0

        # fraction above
        fractions_above = self.fractions_above[t0: tf]
        fractions_above -= correction
        fractions_above[fractions_above<0] = 0

        below = self.integrate(t_normalized, fractions_below)
        above = self.integrate(t_normalized, fractions_above)

        return below, above

    def evaluate_threshold(self, comparison_index):
        """
        Evaluate comparison.

        Args:

            comparison_index (int) - time index for comparison

        Returns:

            below (float) - mean fraction of trajectories below the reference

            above (float) - mean fraction of trajectories above the reference

        """

        below = self.fractions_below[comparison_index]
        above = self.fractions_above[comparison_index]

        # apply correction
        correction = (100-self.bandwidth)/2/100
        below -= correction
        above -= correction

        return max(below, 0), max(above, 0)

    def shade_outlying_areas(self,
                             alpha=0.2,
                             reference_color='k',
                             compared_color='k',
                             above_color='r',
                             above_alpha=0.5,
                             below_color='b',
                             below_alpha=0.5,
                             ax=None,
                             show_threshold=False):
        """
        Visualize comparison by shading the region encompassing trajectories that lie below or above all reference trajectories.

        Args:

            alpha (float) - opacity for shaded regions of confidence band

            reference_color (str) - color for reference confidence band

            compared_color (str) - color for compared confidence band

            above_color, below_color (str) - colors for above/below reference

            ax (matplotlib.axes.AxesSubplot) - if None, create figure

            show_threshold (bool) - if True, show threshold definition

        """

        # create figure if axes weren't provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(3, 2))

        # extract bounds for confidence bands
        threshold_index = self.reached_comparison.nonzero()[0][-1]
        tf = self.comparison_index[threshold_index] + 1
        t = self.t[:tf]
        rbounds = (self.lower[:tf], self.upper[:tf])
        cbounds = (self.compared.lower[self.dim][:tf],
                   self.compared.upper[self.dim][:tf])

        # plot confidence band for reference
        ax.fill_between(t, *rbounds, color=reference_color, alpha=alpha)
        ax.plot(t, rbounds[0], '-k')
        ax.plot(t, rbounds[1], '-k')

        # plot confidence band for compared
        ax.fill_between(t, *cbounds, color=compared_color, alpha=alpha)
        ax.plot(t, cbounds[0], '--k')
        ax.plot(t, cbounds[1], '--k')

        # shade regions below reference
        ind_b, lbound_b, ubound_b = self.extract_region_below(rbounds, cbounds)
        for ind in ind_b:
            ax.fill_between(t[ind],
                            lbound_b[ind],
                            ubound_b[ind],
                            color=below_color,
                            alpha=below_alpha)

        # shade regions above reference
        ind_a, lbound_a, ubound_a = self.extract_region_above(rbounds, cbounds)
        for ind in ind_a:
            ax.fill_between(t[ind],
                            lbound_a[ind],
                            ubound_a[ind],
                            color=above_color,
                            alpha=above_alpha)

        # display threshold definition
        if show_threshold:
            self.display_threshold_definition(ax)

        # format axis
        self.format_axis(ax)

    def display_metrics(self, ax, threshold_index=-1):
        """
        Display comparison metrics on axes.

        Args:

            ax (matplotlib.axes.AxesSubplot)

            threshold_index (int) - index of threshold used

        """

        x = ax.get_xlim()[1] - 0.05*ax.get_xticks().ptp()
        y = ax.get_ylim()[1] - 0.05*ax.get_yticks().ptp()

        kw = dict(ha='right', va='top', fontsize=8)
        ax.text(x, y, '{:0.1%} error'.format(self.error[threshold_index]), **kw)
        ax.text(x, y, '\n{:0.1%} above'.format(self.above[threshold_index]), color='r', **kw)
        ax.text(x, y, '\n\n{:0.1%} below'.format(self.below[threshold_index]), color='b', **kw)
