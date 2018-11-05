from math import floor, log10
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.gridspec import GridSpec
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

    def __init__(self, parameters, response, labels=None, density=100):
        """
        Instantiate parameter sweep visualization.

        Args:

            parameters (np.ndarray[float]) - parameter sets, shape (N, P)

            response (np.ndarray[float]) - response values, length N

            labels (list of str) - labels for each parameter

            density (int) - grid density

        """
        self.parameters = parameters
        self.response = response
        self.labels=labels

        # compile heatmaps
        self.compile(density=density)

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

    @classmethod
    def interpolate(cls, x, y, z, density=100):
        """
        Interpolate data onto a regular 2D grid.

        Args:

            x (1D np.ndarray) - x values

            y (1D np.ndarray) - y values

            z (1D np.ndarray) - response values

            density (int) - grid density

        Returns:

            grid (tuple of 2D np.ndarray) - regular XY grid, e.g. (xi, yi)

            colors (2D np.ndarray) - grid color values, e.g. zi

        """

        # remove nans
        x = x[np.isnan(z) == False]
        y = y[np.isnan(z) == False]
        z = z[np.isnan(z) == False]

        # define grid
        xmin = round(np.log10(x.min()), 1)
        xmax = round(np.log10(x.max()), 1)
        ymin = round(np.log10(y.min()), 1)
        ymax = round(np.log10(y.max()), 1)

        # create regular grid
        xi, yi = np.logspace(xmin, xmax, density), np.logspace(ymin, ymax, density)
        xi, yi = np.meshgrid(xi, yi)

        # transform to normalized coordinates
        x_new, xi_new = cls.normalize(x), cls.normalize(xi, x)
        y_new, yi_new = cls.normalize(y), cls.normalize(yi, y)

        # interpolate data
        zi = griddata((x_new, y_new), z, (xi_new, yi_new), method='linear')

        return (xi, yi), zi

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

                    # interpolate onto 2D grid
                    grid, colors = self.interpolate(x, y, z, density=density)

                    # store heatmap
                    self.heatmaps[(i, j)] = (grid, colors)

    @staticmethod
    def draw_heatmap(ax, grid, colors, cmap=None, bad='k', vmin=-1, vmax=1):
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

        """

        # define colormap
        if cmap is None:
            cmap = plt.cm.get_cmap('seismic')
        cmap.set_bad(bad, 1)

        # add data to plot
        colors = np.ma.masked_invalid(colors)
        im = ax.pcolormesh(*grid, colors, cmap=cmap, vmin=vmin, vmax=vmax)

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

    def format_axes(self, show=True, labelsize=6):
        """
        Format axis.

        Args:

            show (bool) - if False, remove axes

            labelsize (int) - tick label size

        """

        # remove all axes
        if not show:
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
            xmin = self.round(np.min(self.parameters[:, j]), 3)
            xmax = self.round(np.max(self.parameters[:, j]), 3)
            ymin = self.round(np.min(self.parameters[:, i+1]), 3)
            ymax = self.round(np.max(self.parameters[:, i+1]), 3)

            # set bounds
            #ax.set_xlim(xmin, xmax), ax.set_ylim(ymin, ymax)

            # label outermost y axes
            if j == 0:

                # labels
                # yticks = np.logspace(round(np.log10(ymin), 1),
                #                      round(np.log10(ymax), 1), num=3, base=10)
                # ax.set_yticks(yticks)
                # ax.get_yaxis().set_major_formatter(formatter)

                # log labels
                yticks = np.logspace(round(np.log10(ymin), 1),
                                     round(np.log10(ymax), 1), num=6, base=10)
                ax.set_yticks(yticks)
                yticklabels = np.round(np.log10(yticks), 1)
                yticklabels = [None]+list(yticklabels[1:-1])+[None]
                ax.set_yticklabels(yticklabels)
                ax.yaxis.set_tick_params(labelsize=labelsize, pad=0, length=1)

                # add y-axis label
                if self.labels is not None:
                    fmt = lambda label: '${:s}$'.format(label)
                    ax.set_ylabel(fmt(self.labels[i+1]))

            else:
                ax.set_yticks([])
                ax.set_yticklabels([])
                #ax.yaxis.set_minor_locator(plt.FixedLocator([]))

            # label outermost x axes
            if i == self.P - 2:

                # labels
                # xticks = np.logspace(round(np.log10(xmin), 1),
                #                      round(np.log10(xmax), 1), num=3, base=10)
                # ax.set_xticks(xticks)
                # ax.get_xaxis().set_major_formatter(formatter)

                # loglabels
                xticks = np.logspace(round(np.log10(xmin), 1),
                                     round(np.log10(xmax), 1), num=6, base=10)
                ax.set_xticks(xticks)
                xticklabels = np.round(np.log10(xticks), 1)
                xticklabels = [None] + list(xticklabels[1:-1]) + [None]
                ax.set_xticklabels(xticklabels, rotation=45, ha='right')
                #ax.xaxis.set_tick_params(rotation=45)
                ax.xaxis.set_tick_params(labelsize=labelsize, pad=0, length=1)

                # add x-axis label
                if self.labels is not None:
                    fmt = lambda label: '${:s}$'.format(label)
                    ax.set_xlabel(fmt(self.labels[j]))

                for label in ax.xaxis.get_ticklabels():
                    label.set_horizontalalignment('center')

            else:
                ax.set_xticks([])
                ax.set_xticklabels([])
                ax.xaxis.set_minor_locator(plt.FixedLocator([]))
                #ax.spines['bottom'].set_linewidth(10)


    def render(self,
               show=True,
               labelsize=6,
               fig_kwargs={},
               heatmap_kwargs={}):
        """
        Render parameter sweep figure.

        Args:

            show (bool) - if False, remove axes

            labelsize (int) - tick label size

            fig_kwargs: keyword arguments for create_figure

            heatmap_kwargs: keyword arguments for draw_heatmap

        """

        # create figure
        fig, axes_dict = self.create_figure(self.P, **fig_kwargs)
        self.fig = fig
        self.axes_dict = axes_dict

        # draw heatmaps
        for ind, ax in enumerate(self.axes):
            grid, colors = self.heatmaps[self.axes_dict[ind]]
            self.draw_heatmap(ax, grid, colors, **heatmap_kwargs)

        # format axes
        self.format_axes(show=show, labelsize=labelsize)
