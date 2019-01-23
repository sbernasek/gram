from math import floor, log10
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from scipy.interpolate import griddata
from ..figures.settings import *


def fexp(f):
    """ Returns exponent of a float <f>. """
    return int(floor(log10(abs(f)))) if f != 0 else 0

def fman(f):
    """ Returns mantissa of a float <f>. """
    return f/10**fexp(f)


class SweepFigure:
    """
    Class for visualizing a parameter sweep of a model.

    Attributes:

        parameters (np.ndarray[float]) - parameter sets, shape (N, P)

        response (np.ndarray[float]) - response values, length N

        labels (list of str) - labels for each parameter

        heatmaps (dict) - {(row, column): (grid, response)} pairs

        fig (matplotlib.figure.Figure)

        axes_dict (dict) - {axes index: (row, column)} pairs

    Properties:

        P (int) - number of parmeters

        N (int) - number of parameter sets

        axes (list) - figure subpanel axes

    """

    def __init__(self, parameters, response, labels, base, delta):
        """
        Instantiate parameter sweep visualization.

        Args:

            parameters (np.ndarray[float]) - parameter sets, shape (N, P)

            response (np.ndarray[float]) - response values, length N

            labels (list of str) - labels for each parameter

            base (np.ndarray[float]) - mean log10 parameter values, length N

            delta (np.ndarray[float]) - half-range of log10 parameter values

        """
        self.parameters = parameters
        self.response = response
        self.labels = labels
        self.base = base
        self.delta = delta

    @property
    def N(self):
        """ Number of parameter sets. """
        return self.parameters.shape[0]

    @property
    def P(self):
        """ Number of parameters. """
        return self.parameters.shape[1]

    @property
    def axes(self):
        """ Figure axes. """
        return self.fig.axes

    @staticmethod
    def round(x, nchars):
        """ Rounds <x> to nearest value with an exponent of <nchars>. """
        #return 10**(round(np.log10(x)/nchars) * nchars)
        mantissa, exponent = fman(x), fexp(x)
        return round(mantissa, nchars) * (10**exponent)

    @staticmethod
    def normalize(values, bg=None):
        """
        Normalize values relative to some background. If no background is provided, this amounts to rescaling the values to a [0, 1) interval.

        Args:

            values (np.ndarray) - array

            bg (np.ndarray) - background values

        Returns:

            rescaled_values (np.ndarray[float])

        """
        if bg is None:
            bg = values
        values = values.astype(np.float)
        rescaled_values = (values - bg.min()) / (bg.max() - bg.min())
        return rescaled_values

    @staticmethod
    def uniform_sample(xmin, xmax, density):
        """ Returns uniform sample between <xmin> and <xmax>. """
        return np.logspace(xmin, xmax, density)

    @classmethod
    def interpolate(cls, x, y, z, density=100, xbounds=None, ybounds=None):
        """
        Interpolate data onto a regular 2D grid.

        Args:

            x (1D np.ndarray) - x values

            y (1D np.ndarray) - y values

            z (1D np.ndarray) - response values

            density (int) - grid density

            xbounds (tuple) - (xmin, xmax) bounds for grid

            ybounds (tuple) - (ymin, ymax) bounds for grid

        Returns:

            grid (tuple of 2D np.ndarray) - regular XY grid, e.g. (xi, yi)

            colors (2D np.ndarray) - grid color values, e.g. zi

        """

        # remove nans
        x = x[np.isnan(z) == False]
        y = y[np.isnan(z) == False]
        z = z[np.isnan(z) == False]

        # define grid
        if xbounds is None:
            xmin = round(np.log10(x.min()), 1)
            xmax = round(np.log10(x.max()), 1)
        else:
            xmin, xmax = xbounds

        if ybounds is None:
            ymin = round(np.log10(y.min()), 1)
            ymax = round(np.log10(y.max()), 1)
        else:
            ymin, ymax = ybounds

        # create regular grid
        xi = cls.uniform_sample(xmin, xmax, density)
        yi = cls.uniform_sample(ymin, ymax, density)
        xi, yi = np.meshgrid(xi, yi)

        # transform to normalized coordinates
        x_new, xi_new = cls.normalize(x), cls.normalize(xi, x)
        y_new, yi_new = cls.normalize(y), cls.normalize(yi, y)

        # interpolate data
        zi = griddata((x_new, y_new), z, (xi_new, yi_new), method='linear')

        return (xi, yi), zi

    def get_bounds(self, i, j):
        """ Returns x/y bounds for row <i> col <j> """
        base = self.base[[j, i+1]]
        (xmin, ymin), (xmax, ymax) = base-self.delta, base+self.delta
        return (xmin, xmax), (ymin, ymax)

    def compile(self, density=100):
        """
        Compile heatmaps.

        Args:

            density (int) - grid density

        """

        self.heatmaps = {}

        # iterate through each subpanel
        for i in range(self.P - 1):
            for j in range(self.P - 1):
                if j <= i:

                    # get values
                    x = self.parameters[:, j]
                    y = self.parameters[:, i+1]
                    z = self.response

                    # get bounds
                    xbounds, ybounds = self.get_bounds(i, j)

                    # interpolate onto 2D grid
                    grid, colors = self.interpolate(x, y, z,
                                                    density=density,
                                                    xbounds=xbounds,
                                                    ybounds=ybounds)

                    # store heatmap
                    self.heatmaps[(i, j)] = (grid, colors)

    @staticmethod
    def draw_heatmap(ax, grid, colors, cmap=None, bad='k', vmin=-1, vmax=1, rasterized=True):
        """
        Draw heatmap on specified axes.

        Args:

            ax (matplotlib.axes.AxesSubplot) - axis object

            grid (tuple of 2D np.ndarray) - regular XY grid, e.g. (xi, yi)

            colors (2D np.ndarray) - grid color values, e.g. zi

            cmap (matplotlib.colormaps.ColorMap) - colormap for patches

            bad (str) - color for NaN patches

            vmin (float) - lower bound for color scale

            vmax (float) - upper bound for color scale

            rasterized (bool) - if True, rasterize mesh

        """

        # define colormap
        if cmap is None:
            cmap = plt.cm.get_cmap('seismic')
        cmap.set_bad(bad, 1)

        # add data to plot
        colors = np.ma.masked_invalid(colors)
        im = ax.pcolormesh(*grid, colors, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=rasterized)

    @staticmethod
    def create_figure(num_params, gap=0.05, figsize=(5, 5)):
        """
        Create figure for parameter sweep.

        Args:

            num_params (int) - number of parameters

            gap (float) - spacing between panels

            figsize (tuple) - figure size

        Returns:

            fig (matplotlib.figures.Figure)

            axes_dict (dict) - {axes index: (row, column)} pairs

        """

        # create figure
        fig = plt.figure(figsize=figsize, frameon=False)

        # add subplots
        dim = num_params - 1
        gs = GridSpec(nrows=dim, ncols=dim, wspace=gap, hspace=gap)
        xx, yy = np.tril_indices(dim, k=0)
        axes_dict = {}
        for ind, (i, j) in enumerate(zip(xx.ravel(), yy.ravel())):
            fig.add_subplot(gs[i, j])
            axes_dict[ind] = (i, j)
        return fig, axes_dict

    def format_axes(self, include_axis=True, include_labels=True, labelsize=6):
        """
        Format axis.

        Args:

            include_axis (bool) - if False, remove axes

            include_labels (bool) - if False, remove axis labels

            labelsize (int) - tick label size

        """

        # remove all axes
        if not include_axis:
            for ax in self.axes:
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

        # define tick formatter
        f = lambda n, _: '$\mathregular{10^{%.1f}}$' % np.log10(n)
        formatter = mtick.FuncFormatter(f)

        # format used axes
        for ind, ax in enumerate(self.axes):

            # get row/column indices
            i, j = self.axes_dict[ind]

            # log scale axes
            ax.set_xscale('log', subsx=[])
            ax.set_yscale('log', subsy=[])
            ax.minorticks_off()

            # get bounds of sampled data
            xbounds, ybounds = self.get_bounds(i, j)
            btick = ['']

            # label outermost y axes
            if j == 0:

                yticks = np.logspace(*ybounds, num=6, base=10)
                ax.set_yticks(yticks)
                yticklabels = np.round(np.log10(yticks), 1)
                yticklabels = btick+list(yticklabels[1:-1])+btick
                ax.set_yticklabels(yticklabels)
                ax.yaxis.set_tick_params(labelsize=labelsize, pad=0, length=1)

                # add y-axis label
                if self.labels is not None and include_labels:
                    prefix = r'$\log_{10}\ $'
                    fmt = lambda label: prefix + '${:s}$'.format(label)
                    ax.set_ylabel(fmt(self.labels[i+1]))

            else:
                ax.set_yticks([])
                ax.set_yticklabels([])

            # label outermost x axes
            if i == self.P - 2:

                xticks = np.logspace(*xbounds, num=6, base=10)
                ax.set_xticks(xticks)
                xticklabels = np.round(np.log10(xticks), 1)
                xticklabels = btick + list(xticklabels[1:-1]) + btick
                ax.set_xticklabels(xticklabels, rotation=45, ha='right')
                #ax.xaxis.set_tick_params(rotation=45)
                ax.xaxis.set_tick_params(labelsize=labelsize, pad=0, length=1)

                # add x-axis label
                if self.labels is not None and include_labels:
                    prefix = r'$\log_{10}\ $'
                    fmt = lambda label: prefix + '${:s}$'.format(label)
                    ax.set_xlabel(fmt(self.labels[j]))

                for label in ax.xaxis.get_ticklabels():
                    label.set_horizontalalignment('center')

            else:
                ax.set_xticks([])
                ax.set_xticklabels([])
                ax.xaxis.set_minor_locator(plt.FixedLocator([]))

    def render(self,
               density=100,
               include_axis=True,
               include_labels=True,
               labelsize=6,
               fig_kwargs={},
               heatmap_kwargs={}):
        """
        Render parameter sweep figure.

        Args:

            density (int) - grid density

            include_axis (bool) - if False, remove axes

            include_labels (bool) - if False, remove axis labels

            labelsize (int) - tick label size

            fig_kwargs: keyword arguments for create_figure

            heatmap_kwargs: keyword arguments for draw_heatmap

        """

        # compile heatmaps
        self.compile(density=density)

        # create figure
        fig, axes_dict = self.create_figure(self.P, **fig_kwargs)
        self.fig = fig
        self.axes_dict = axes_dict

        # draw heatmaps
        for ind, ax in enumerate(self.axes):
            grid, colors = self.heatmaps[self.axes_dict[ind]]
            self.draw_heatmap(ax, grid, colors, **heatmap_kwargs)

        # format axes
        self.format_axes(include_axis=include_axis, include_labels=include_labels, labelsize=labelsize)


class LinearSweepFigure(SweepFigure):
    """
    Class for visualizing a parameter sweep of growth dependencies.
    """

    @staticmethod
    def uniform_sample(xmin, xmax, density):
        """ Returns uniform sample between <xmin> and <xmax>. """
        return np.linspace(xmin, xmax, density)

    def format_axes(self, include_axis=True, include_labels=True, labelsize=6):
        """
        Format axis.

        Args:

            include_axis (bool) - if False, remove axes

            include_labels (bool) - if False, remove axis labels

            labelsize (int) - tick label size

        """

        # remove all axes
        if not include_axis:
            for ax in self.axes:
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

        # define tick formatter
        f = lambda n, _: '{:0.1f}'.format(n)
        formatter = mtick.FuncFormatter(f)

        # format used axes
        for ind, ax in enumerate(self.axes):

            # get row/column indices
            i, j = self.axes_dict[ind]

            # log scale axes
            ax.minorticks_off()

            # get bounds of sampled data
            xbounds, ybounds = self.get_bounds(i, j)

            # label outermost y axes
            if j == 0:

                yticks = np.linspace(*ybounds, num=5)
                ax.set_yticks(yticks)
                yticklabels = np.round(yticks, 1)
                #yticklabels = [None]+list(yticklabels[1:-1])+[None]
                #ax.set_yticklabels(yticklabels)
                ax.yaxis.set_tick_params(labelsize=labelsize, pad=1, length=3)

                # add y-axis label
                if self.labels is not None and include_labels:
                    fmt = lambda label: r'{:s} / {:s}'.format('log({:s})'.format(label), 'log(Growth)')
                    ax.set_ylabel(fmt(self.labels[i+1]), fontsize=labelsize+1)

            else:
                ax.set_yticks([])
                ax.set_yticklabels([])

            # label outermost x axes
            if i == self.P - 2:

                xticks = np.linspace(*xbounds, num=5)
                ax.set_xticks(xticks)
                xticklabels = np.round(xticks, 1)
                #xticklabels = [None] + list(xticklabels[1:-1]) + [None]
                #ax.set_xticklabels(xticklabels, rotation=45, ha='right')
                #ax.xaxis.set_tick_params(rotation=45)
                ax.xaxis.set_tick_params(labelsize=labelsize, pad=1, length=3)

                # add x-axis label
                if self.labels is not None and include_labels:
                    fmt = lambda label: r'{:s} / {:s}'.format('log({:s})'.format(label), 'log(Growth)')
                    ax.set_xlabel(fmt(self.labels[j]), fontsize=labelsize+1)

                for label in ax.xaxis.get_ticklabels():
                    label.set_horizontalalignment('center')

            else:
                ax.set_xticks([])
                ax.set_xticklabels([])
                ax.xaxis.set_minor_locator(plt.FixedLocator([]))


class SweepHistogram:
    """
    Class for visualizing the results of a parameter sweep of a model using a one dimensional histogram.

    Attributes:

        values (np.ndarray[float]) - values, length N

        fig (matplotlib.figure.Figure)

    Properties:

        N (int) - number of parameter sets

        ax (list) - axis

    """

    def __init__(self, values):
        """
        Instantiate parameter sweep visualization.

        Args:

            values (np.ndarray[float]) - values, length N

        """
        self.values = values

    def save(self, fname, fmt='pdf', dpi=300, transparent=True):
        """ Save figure as <fname>. """
        self.fig.savefig(fname, format=fmt, dpi=dpi, transparent=transparent)

    @property
    def N(self):
        """ Number of response values. """
        return self.values.shape[0]

    @staticmethod
    def create_figure(figsize=(2, 1.25)):
        """
        Create figure for parameter sweep.

        Args:

            figsize (tuple) - figure size

        Returns:

            fig (matplotlib.figures.Figure)

        """
        fig, ax = plt.subplots(figsize=figsize, frameon=False)
        return fig, ax

    def render(self,
               bins=20,
               vlim=(-1, 1),
               xlim=(-1, 1),
               cmap=None,
               log=True,
               include_labels=True,
               labelsize=7,
               fig_kwargs={}):
        """
        Render parameter sweep figure.

        Args:

            bins (int or list) - histogram bins

            vlim (tuple) - lower and upper bounds for colormap

            xlim (tuple) - lower and upper bounds for response values

            cmap (matplotlib.colormap) - colormap for bar facecolor

            include_axis (bool) - if False, remove axes

            include_labels (bool) - if False, remove axis labels

            labelsize (int) - tick label size

            fig_kwargs: keyword arguments for create_figure

        """

        # create figure
        fig, ax = self.create_figure(**fig_kwargs)
        self.fig = fig
        self.ax = ax

        # plot histogram
        self.vlim = vlim
        self.xlim = xlim
        self.plot(bins, cmap)

        # format axes
        self.format_axes(log=log, include_labels=include_labels, labelsize=labelsize)

    def plot(self, bins=50, cmap=None, rasterized=True):
        """
        Plot histogram on axes.

        Args:

            bins (int or array like) - bins for histogram

            cmap (matplotlib.colormap) - colormap for bar facecolor

            rasterized (bool) - if True, rasterize bars

        """

        if cmap is None:
            cmap = plt.cm.seismic

        # plot histogram
        bins = np.linspace(*self.xlim, num=bins)
        counts, edges, patches = self.ax.hist(self.values,
                                     bins=bins,
                                     density=False,
                                     color='k')
        _ = self.ax.hist(self.values,
                         bins=bins,
                         density=False,
                         color='k',
                         lw=0.25,
                         histtype='step',
                         rasterized=rasterized)
        bin_centers = 0.5 * (edges[:-1] + edges[1:])

        # color bars
        norm = Normalize(*self.vlim)
        for c, p in zip(bin_centers, patches):
            plt.setp(p, 'facecolor', cmap(norm(c)))

    def format_axes(self, log=True, include_labels=True, labelsize=7):
        """
        Format histogram axis.

        Args:

            log (bool) - if True, logscale y axis

            include_labels (bool) - if False, remove labels

            labelsize (int) - label font size

        """

        # set axis limits and scale
        self.ax.set_xlim(*self.xlim)
        if log:
            self.ax.set_yscale('symlog', linthreshy=1)

        # format axes
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['top'].set_visible(False)

        # label axes
        if include_labels:
            self.ax.set_ylabel('Num. parameter sets', fontsize=labelsize)
            self.ax.set_xticklabels(['{:.0%}'.format(x) for x in self.ax.get_xticks()], fontsize=labelsize)
        else:
            self.ax.set_yticks([])
            self.ax.set_xticks([])


class SweepLines(SweepHistogram):
    """
    Class for visualizing the results of a parameter sweep as a function of where the success threshold is set.

    Attributes:

        values (np.ndarray[float]) - values, length N

        thresholds (np.ndarray[float]) - fractions of peak level

        fig (matplotlib.figure.Figure)

    Properties:

        N (int) - number of parameter sets

        ax (list) - axis

    """

    def __init__(self, values, thresholds):
        """
        Instantiate parameter sweep visualization.

        Args:

            values (np.ndarray[float]) - values, length N

            thresholds (np.ndarray[float]) - fractions of peak level

        """
        self.values = values
        self.thresholds = thresholds

    def render(self,
               vlim=(-1, 1),
               ylim=(-1, 1),
               cmap=None,
               include_labels=True,
               labelsize=7,
               fig_kwargs={}):
        """
        Render parameter sweep figure.

        Args:

            vlim (tuple) - lower and upper bounds for colormap

            ylim (tuple) - lower and upper bounds for response values

            cmap (matplotlib.colormap) - colormap for bar facecolor

            include_axis (bool) - if False, remove axes

            include_labels (bool) - if False, remove axis labels

            labelsize (int) - tick label size

            fig_kwargs: keyword arguments for create_figure

        """

        # create figure
        fig, ax = self.create_figure(**fig_kwargs)
        self.fig = fig
        self.ax = ax

        # plot histogram
        self.vlim = vlim
        self.ylim = ylim
        self.plot(cmap)

        # format axes
        self.format_axes(include_labels=include_labels, labelsize=labelsize)

    def plot(self, cmap=None, rasterized=True):
        """
        Plot histogram on axes.

        Args:

            cmap (matplotlib.colormap) - colormap for bar facecolor

            rasterized (bool) - if True, rasterize lines

        """

        if cmap is None:
            cmap = plt.cm.OrRd

        # sort line order by difference over range
        ptp = np.nanmax(self.values, axis=1) - np.nanmin(self.values, axis=1)
        ptp /= np.nanmax(ptp)
        sort_ind = np.argsort(ptp)
        values = self.values[sort_ind]

        # compile lines
        thresholds = np.tile(self.thresholds, (self.values.shape[0], 1))
        line_data = np.swapaxes(np.stack((thresholds, values)).T, 0, 1)
        lines = LineCollection(line_data,
                               linewidth=0.5,
                               alpha=1.,
                               colors=cmap(ptp[sort_ind]),
                               rasterized=rasterized)

        # add lines to axis
        self.ax.add_collection(lines)

    def format_axes(self, include_labels=True, labelsize=7):
        """
        Format histogram axis.

        Args:

            include_labels (bool) - if False, remove labels

            labelsize (int) - label font size

        """

        # set axis limits and scale
        self.ax.invert_xaxis()
        self.ax.set_ylim(*self.ylim)
        self.ax.set_xlim(max(self.thresholds), min(self.thresholds))
        self.ax.set_xticks(self.thresholds)

        # format axes
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['top'].set_visible(False)

        # label axes
        if include_labels:
            self.ax.set_ylabel('Error frequency', fontsize=labelsize)
            self.ax.set_xlabel('Threshold position\n(fraction of peak mean value)', fontsize=labelsize)
            self.ax.set_xticklabels(['{:0.1f}'.format(x) for x in self.ax.get_xticks()], fontsize=labelsize)
            self.ax.set_yticklabels(['{:2.0%}'.format(x) for x in self.ax.get_yticks()], fontsize=labelsize)
        else:
            self.ax.set_yticks([])
            self.ax.set_xticks([])
