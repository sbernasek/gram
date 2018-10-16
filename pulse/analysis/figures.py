__author__ = 'Sebi'

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np
import os


def generate_heatmap(grids, scheme='error', controllers=None, conditions=None, axis_labels=True, color_bar=True):
    """
    Creates heatmaps for each condition in which the grid is defined by first and second controllers, while intensities
    correspond to the penetrance of mutant phenotype.

    Parameters:
        grids (dict) - keys are growth conditions, values are N x N maskedarrays of population overlap scores, where N
        is the number of controllers considered.
        scheme (str) - colorbar scheme to use (error or delta)
        controllers (list) - list of strings corresponding to controllers used to define heatmap dimensions
        conditions (list) - list of conditions for which heatmaps are generated
        axis_labels (bool) - if False, remove all axes and associated labels
        color_bar (bool) - if True, include colorbar

    Returns:
        figs (list) - figures
    """

    # create color map with masked values for controller pairings that never reached threshold
    if scheme == 'delta':
        cmap = plt.cm.PuOr_r
        vmin, vmax = -1, 1
    else:
        cmap = plt.cm.copper_r
        vmin, vmax = 0, 1
    cmap.set_bad(color='0.7', alpha=1)

    if controllers is None:
        controllers = ('gene', 'transcript', 'protein')

    if conditions is None:
        conditions = [condition for condition in grids.keys()]

    # generate heat map
    figs = []
    for condition in conditions:

        # get data and create axes
        data = grids[condition]

        # create heatmap
        fig, ax = plt.subplots(figsize=(12, 15))
        fig.suptitle(condition, x=0, y=1., fontsize=48, fontweight='bold', horizontalalignment='center')
        heatmap = ax.pcolormesh(data, cmap=cmap, edgecolors='None', vmin=vmin, vmax=vmax)

        # annotate null values with NA (note transpose for failure labels)
        failures = np.where(data.mask == True)
        for y, x in zip(*failures):
            ax.text(x+0.5, y+0.5, 'NA', fontsize=24, fontweight='bold', color='k', horizontalalignment='center', verticalalignment='center')

        # format table (flip and invert axes), set tick spacing
        ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)
        ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)
        ax.set_xticks(np.arange(data.shape[1]), minor=True)
        ax.set_yticks(np.arange(data.shape[0]), minor=True)
        ax.invert_yaxis()
        ax.xaxis.tick_top()
        ax.axis('tight')

        # format heatmap tick labels
        if axis_labels is True:
            column_labels, row_labels = controllers, controllers
            ax.set_xticklabels(column_labels, minor=False, fontsize=16, rotation=0)
            plt.xticks(rotation=30, horizontalalignment='left')
            ax.set_yticklabels(row_labels, minor=False, fontsize=16)
        else:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            fig.suptitle("")

        # format heatmap tick marks
        plt.tick_params(
            axis='both',
            which='both',
            bottom='off',
            top='off',
            right='off',
            left='off')

        # add colorbar and format colorbar ticks
        if color_bar:
            color_bar = plt.colorbar(heatmap, orientation='horizontal')
            #color_bar.set_label('phenotype penetrance', fontsize=36)
            color_bar.set_alpha(1)
            color_bar.draw_all()

            # set bounds for colorbar
            if scheme == 'delta':
                color_bar.set_ticks(np.arange(-1, 1.1, 0.5))
                color_bar.set_ticklabels(["-100%", "-50%", "0%", "+50%", "+100%"])
            else:
                color_bar.set_ticks(np.arange(0, 1.1, 0.2))
                color_bar.set_ticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])

            color_bar.ax.tick_params(labelsize=28)
        else:
            fig.set_size_inches(12, 12)

        figs.append(fig)

    return figs


def generate_bar_plot(scores, commitment_index=2):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))

    for i, repressor1 in enumerate(('gene', 'transcript', 'protein')):
        for j, repressor2 in enumerate(('gene', 'transcript', 'protein')):

            # get scores and parse scores/labels
            score_dict = scores[(repressor1, repressor2)]

            labels, data = zip(*[(condition, score_dict[condition]) for condition in ('normal', 'diabetic', 'minute')])

            # raise error if conditions are out of order
            if sum([a!=b for a, b in zip(labels, ['normal', 'diabetic', 'minute'])]) > 0:
                raise ValueError('Conditions out of order.')

            ax = axes[i, j]
            width = 0.5
            ind = np.arange(3)

            if type(data[0]) is float:
                heights = [x*100 if x is not None else 0 for x in data]
            else:
                heights = [x[commitment_index]*100 if x[commitment_index] is not None else 0 for x in data]
            patches = ax.bar(ind, heights, width)
            patches[0].set_color('darkseagreen')
            patches[1].set_color('thistle')
            patches[2].set_color('indianred')
            ax.set_ylim(0, 100)

            if j != 0:
                ax.set_yticks([])

            if j == 0:
                ax.set_ylabel(repressor1 + '\n\n Error Frequency', fontsize=18)
                ax.set_yticks([0, 25, 50, 75, 100])
                fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
                ytick_formatter = mtick.FormatStrFormatter(fmt)
                ax.yaxis.set_major_formatter(ytick_formatter)

            if i == 0:
                ax.set_title(repressor2, fontsize=18)

            ax.set_xticks([])
            ax.tick_params(labelsize=18)
            ax.set_xlim(-width, 3)

            if i == 0 and j == 0:
                p1 = mpatches.Patch(facecolor='darkseagreen', linewidth=0, label='normal')
                p2 = mpatches.Patch(facecolor='thistle', linewidth=0, label='slow metabolism')
                p3 = mpatches.Patch(facecolor='indianred', linewidth=0, label='slow translation')
                ax.legend(handles=[p1, p2, p3], loc=2, frameon=False, fontsize=16)

    plt.tight_layout()


def generate_ef_threshold_sensitivity(data, condition='normal', colorbar=False):
    """
    Generates plot with error frequency vs commitment threshold contours for the condition specified.

    Parameters:
        data (SweepData object) - simulation results
        condition (str) - growth condition

    Returns:
        fig (matplotlib figure)
    """

    # get data for current condition
    error_frequencies = data.error_frequencies[condition]

    # set colormap
    vmin, vmax = 0, 1
    cmap = plt.get_cmap('OrRd')
    cNorm = colors.Normalize(vmin=vmin, vmax=vmax)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

    # define figure
    scale = 0.7
    fig, ax = plt.subplots(ncols=1, figsize=(8*scale, 5*scale))

    # define threshold fractions
    threshold_fractions = np.arange(0.1, 1, 0.1)

    # plot error frequency samples, with line color and zorder scaled by the sample range (puts greatest change on top)
    ef_range = np.nanmax(error_frequencies, axis=1)-np.nanmin(error_frequencies, axis=1)
    ef_sort_indices = np.argsort(ef_range)
    for sample, delta in zip(error_frequencies[ef_sort_indices, :], ef_range[ef_sort_indices]):
        colorVal =  scalarMap.to_rgba(delta)
        im = ax.plot(threshold_fractions, sample*100, linewidth=1, color=colorVal)

    # format axes
    ax.set_ylabel('Error Frequency', fontsize=16)
    ax.set_ylim(-10, 110)
    ax.set_xlabel('Success Threshold\n (fraction of peak mean value)', fontsize=16)
    ax.set_xlim(0, 1)

    # format tick labels
    x_fmt = '%.1f%'
    y_fmt = '%.0f%%'
    ax.set_xticks([0.2, 0.4, 0.6, 0.8])
    ytick_fmt = mtick.FormatStrFormatter(y_fmt)
    ax.yaxis.set_major_formatter(ytick_fmt)
    ax.tick_params(labelsize=16, pad=10)

    plt.tight_layout()

    # add colorbar
    if colorbar is True:
        scalarMap.set_array(ef_range)
        cbar = plt.colorbar(scalarMap, ax=ax, orientation='vertical', fraction=0.05, pad=0.05, aspect=12)
        cbar.set_label('Range of EF \n(max - min)', fontsize=16)
        cbar.set_ticks(np.linspace(vmin, vmax, 5))
        cbar.set_ticklabels([str(int(round(100*lab, 2)))+'%' for lab in np.linspace(vmin, vmax, 5)])
        cbar.ax.tick_params(labelsize=16)

    return fig


def generate_suppression_threshold_sensitivity(data):
    """
    Generates plot with error suppression vs commitment threshold contours for the condition specified.

    Parameters:
        data (SweepData object) - simulation results

    Returns:
        fig (matplotlib figure)
    """

    # get data
    diabetic = data.error_frequencies['normal'] - data.error_frequencies['diabetic']
    minute = data.error_frequencies['normal'] - data.error_frequencies['minute']

    # set colormap
    vmin, vmax = 0, 1
    cmap = plt.get_cmap('OrRd')
    cNorm = colors.Normalize(vmin=vmin, vmax=vmax)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

    scale = 0.7
    fig, axes = plt.subplots(ncols=2, figsize=(16*scale, 6*scale), sharey=True)
    ax0, ax1 = axes

    fractions = np.arange(0.1, 1, 0.1)

    # plot diabetic samples, with line color and zorder scaled by the sample range
    diabetic_range = np.nanmax(diabetic, axis=1)-np.nanmin(diabetic, axis=1)
    diabetic_sort_indices = np.argsort(diabetic_range)
    for sample, delta in zip(diabetic[diabetic_sort_indices, :], diabetic_range[diabetic_sort_indices]):
        colorVal =  scalarMap.to_rgba(delta)
        im0 = ax0.plot(fractions, sample*100, linewidth=1, color=colorVal)

    # plot minute samples, with line color and zorder scaled by the sample range
    minute_range = np.nanmax(minute, axis=1)-np.nanmin(minute, axis=1)
    minute_sort_indices = np.argsort(minute_range)
    for sample, delta in zip(minute[minute_sort_indices, :], minute_range[minute_sort_indices]):
        colorVal =  scalarMap.to_rgba(delta)
        im1 = ax1.plot(fractions, sample*100, linewidth=1, color=colorVal)

    # add zero line
    for ax in axes:
        ax.plot([0.1, 0.9], [0, 0], '--k', linewidth=3)

    # format plot
    ax0.set_title('Reduced Metabolism', fontsize=18)
    ax1.set_title('Reduced Protein Synthesis', fontsize=18)

    ax0.set_ylabel('Decrease in Error Frequency', fontsize=16)
    ax0.set_ylim(-20, 110)
    for ax in axes:
        ax.set_xlabel('Success Threshold\n (fraction of peak mean value)', fontsize=16)
        ax.set_xlim(0, 1)

        # format tick labels
        x_fmt = '%.1f%'
        y_fmt = '%.0f%%'
        ax.set_xticks([0.2, 0.4, 0.6, 0.8])
        #xtick_fmt = ticker.FormatStrFormatter(x_fmt)
        ytick_fmt = mtick.FormatStrFormatter(y_fmt)
        #ax.xaxis.set_major_formatter(xtick_fmt)
        ax.yaxis.set_major_formatter(ytick_fmt)
        ax.tick_params(labelsize=16, pad=10)

    plt.tight_layout()

    # add colorbar
    scalarMap.set_array(diabetic_range)
    cbar = plt.colorbar(scalarMap, ax=list(axes), orientation='vertical', fraction=0.05, pad=0.025, aspect=10)
    cbar.set_label('Difference between max and min \ndecrease in error frequency', fontsize=16)
    cbar.set_ticks(np.linspace(vmin, vmax, 5))
    cbar.set_ticklabels([str(int(round(100*lab, 2)))+'%' for lab in np.linspace(vmin, vmax, 5)])
    cbar.ax.tick_params(labelsize=16)

    return fig


def save_figure(figure, name, path='./graphics'):
    """
    Save figure in EPS format.

    Parameters:
        figure (matplotlib figure)
        name (str) - figure file name
        path (str) - directory to which figure is to be saved
    """

    figure.savefig(os.path.join(path, name + '.eps'), format='eps', dpi=100)

def save_figures(figures, names, path='./graphics'):
    """ Save a list of figures in EPS format. """
    _ = [save_figure(fig, name, path=path) for fig, name in zip(figures, names)]











