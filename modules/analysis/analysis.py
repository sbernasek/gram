__author__ = 'Sebi'

from nevosim.solver.signals import cSquarePulse
from nevosim.timeseries import GaussianModel
from modules.parameters import time_scaling, input_sensitivity
from modules.data_handling import load
import numpy as np
import matplotlib.pyplot as plt
import os
import glob


def generate_time_series_figure(wildtype_model, mutant_model, condition=None, commitment_times=None, thresholds=None, error_frequencies=None, dim=1, include_samples=False, ax=None, fill_alpha=0.5):
    """
    Generates figure containing confidence intervals for wildtype and mutant time series.

    Parameters:
        wildtype_model (TimeSeriesModel) - wildtype time series
        mutant_model (TimeSeriesModel) - mutant time series
        dim (int) - dimension of state space
        include_samples (bool) - if True, add individual samples to plot
        ax (matplotlib axes object)
        fill_alpha (float) - alpha value for fill
    """

    # create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))

    # plot confidence intervals
    mutant_model.plot_confidence_interval(ax=ax, dims=(dim,), colors=('red',), time_scaling=time_scaling, alpha=fill_alpha)
    wildtype_model.plot_confidence_interval(ax=ax, dims=(dim,), colors=('blue',), time_scaling=time_scaling, alpha=fill_alpha)

    # plot individual samples
    if include_samples is True:
        mutant_model.plot_samples(ax=ax, dims=(dim,), colors=('red',), time_scaling=time_scaling)
        wildtype_model.plot_samples(ax=ax, dims=(dim,), colors=('blue',), time_scaling=time_scaling)

    if commitment_times is not None and thresholds is not None:
        if type(commitment_times) is float:
            time = commitment_times
            threshold = thresholds
        else:
            time = commitment_times[2]
            threshold = thresholds[2]

        # add threshold (TO DO: need to account for array)
        times_post_threshold = np.arange(time*time_scaling, wildtype_model.t[-1]*time_scaling, 1)
        ax.plot(times_post_threshold, threshold*np.ones(len(times_post_threshold)), '--k', linewidth=3)
        ax.plot(time*time_scaling, threshold, '.k', markersize=15)

    if type(error_frequencies) is float:
        ax.set_title('{:s} \n Error Frequency: {:0.2%}'.format(str(condition), error_frequencies), fontsize=14)
    elif error_frequencies is not None:
        ax.set_title('{:s} \n Error Frequency: {:0.2%}'.format(str(condition), error_frequencies[2]), fontsize=14)
    else:
        ax.set_title('{:s} \n Failed to Commit'.format(str(condition)), fontsize=14)

def evaluate_error_frequency(wildtype, mutant, condition=None, input_scaling=False, num_trials=1000, dt=1, ic=None, basal_input=0., input_start=25, input_duration=3, duration=100, normalization=None, method='cy_hybrid', ax=None, plot=False, plot_samples=False, fill_alpha=0.5):
    """
    Determine error frequency under a single environmental condition.

    Parameters:
        wildtype, mutant (FeedbackSystem objects)
    """

    # get input scaling
    duration_scaling, magnitude_scaling = 1, 1
    if input_scaling is True:
        duration_scaling = input_sensitivity[condition]['duration']
        magnitude_scaling = input_sensitivity[condition]['magnitude']

    # instantiate cythonized input signal
    t_on, t_off = input_start/time_scaling, (input_start+input_duration*duration_scaling)/time_scaling
    input_signal = cSquarePulse(t_on, t_off, off=basal_input, on=magnitude_scaling)

    # scale duration (from hours to minutes)
    duration /= time_scaling

    # use method of moments
    if method == 'moments':

        # handles vectorized input signal
        def input_signal(t):
            t = np.array(t)
            input_ = basal_input*np.ones(len(t))
            input_[np.logical_and(t > (input_start/time_scaling), t < (input_start+input_duration)/time_scaling)] = 1
            return input_

        wildtype_model = wildtype.run_analytical_simulation(input_signal=input_signal, condition=condition, ic=ic, dt=dt, duration=duration, normalization=normalization, constrain_positive=True)

        mutant_model = mutant.run_analytical_simulation(input_signal=input_signal, condition=condition, ic=ic, dt=dt, duration=duration, normalization=normalization, constrain_positive=True)

    # use montecarlo methods
    else:

        # run wild type simulations
        _, wildtype_model = wildtype.run_stochastic_simulation(input_signal, ic=ic, condition=condition, normalization=normalization, num_trials=num_trials, dt=dt, duration=duration, method=method)

        _, mutant_model = mutant.run_stochastic_simulation(input_signal, ic=ic, condition=condition, normalization=normalization, num_trials=num_trials, dt=dt, duration=duration, method=method)

    # set wildtpye input level
    if basal_input != 0.:
        wt_ic = wildtype_model.mean[:, -1]
    else:
        wt_ic = np.zeros(wildtype_model.mean.shape[0])

    # get failure frequency at commitment time
    times, thresholds, error_frequencies = get_error_frequency(wildtype_model, mutant_model, wt_ic=wt_ic, normalization=normalization, dim=wildtype.output_node)

    # plot trajectories
    if plot is True:
        generate_time_series_figure(wildtype_model, mutant_model, condition=condition, commitment_times=times, thresholds=thresholds, error_frequencies=error_frequencies, dim=wildtype.output_node, include_samples=plot_samples, fill_alpha=fill_alpha, ax=ax)

    return error_frequencies, wildtype_model, mutant_model


def get_error_frequency(wildtype_model, mutant_model, wt_ic=None, normalization=None, fraction=None, dim=None, percentile=99):
    """
    Given a wildtype and mutant TimeSeriesModel, compute the error frequency (complement to population overlap).

    Parameters:
        wildtype_model, mutant_model (TimeSeriesModel) - time series models
        wt_ic (float) - initial condition used for normalization
        normalization (str) - if 'relative', divide by initial condition, if 'deviation' substract initial condition
        fraction (float) - fraction of maximum mean wildtype expression at which error frequency is evaluated
        dim (int) - dimension of state space used to determine error frequency
    """

    # if no fraction is specified, use several
    if fraction is None:
        fraction = np.arange(0.1, 1, 0.1)

    if dim is None:
        dim = 1

    # determine population overlaps
    success_thresholds = wildtype_model.get_percentile(percentile=percentile)[dim]
    success_frequencies = mutant_model.evaluate_cdf(dim, success_thresholds)

    # find commitment time
    if normalization == 'relative':
        commitment_threshold = 1 + fraction * (wildtype_model.peaks[dim]-1)
    elif normalization == 'deviation':
        commitment_threshold = fraction * wildtype_model.peaks[dim]
    else:
        commitment_threshold = wt_ic[dim] + fraction * (wildtype_model.peaks[dim] - wt_ic[dim])

    # if one fraction is specified, compute failure frequency
    if type(commitment_threshold) is np.float64:
        commitment_index, commitment_time = wildtype_model.get_time_of_value(dim=dim, percentile=percentile, value=commitment_threshold, after_peak=True)

        # get error frequencies
        failure_frequency = 1-success_frequencies[commitment_index]

        # if wildtype failed to reach threshold, set failure frequency to NAN
        if commitment_time == wildtype_model.t[-1]:
            failure_frequency = np.nan

    # if multiple fractions are specified, compute array of failure frequencies
    else:
        commitment_time, commitment_index = [], []
        for threshold in commitment_threshold:
            index, time = wildtype_model.get_time_of_value(dim=dim, percentile=percentile, value=threshold, after_peak=True)
            commitment_time.append(time), commitment_index.append(index)
        commitment_time = np.array(commitment_time)
        commitment_index = np.array(commitment_index)

        # get error frequencies
        failure_frequency = 1-success_frequencies[commitment_index]

        # if wildtype failed to reach threshold, set failure frequency to NAN
        failed_to_commit = np.where(commitment_time==wildtype_model.t[-1])
        failure_frequency[failed_to_commit] = np.nan

    return commitment_time, commitment_threshold, failure_frequency


def load_scores(data_path):
    """
    Load scores from existing simulation ran on cluster.
    """

    scores = {}

    # iterate over each simulation
    for i, job in enumerate(glob.glob(data_path+'/*')):

        # get repressors and replicate number
        job_name = job.split('/')[-1]
        if '_' in job_name:
            job_name, replicate = job_name.split('_')
        repressor1, repressor2 = job_name.split('-')

        # load data
        error_frequencies_path = os.path.join(job, 'error_frequencies.json')
        error_frequencies = load(error_frequencies_path)

        # use repressor pairing as key
        key = (repressor1, repressor2)

        # if repressor pairing has not been added to overall results, add it
        if key not in scores.keys():
            scores[key] = {condition: [score] for condition, score in error_frequencies.items()}

        # otherwise, append score to existing results
        else:
            for condition, score in error_frequencies.items():
                scores[key][condition].append(score)

    return scores


def get_scores_for_deviations(data_path, dim=1, input_start=5):
    """
    Recompiles scores based on deviations from steady state.
    """

    scores_for_each_pairing = {}

    # iterate over each simulation
    for i, job in enumerate(glob.glob(data_path + '/*')):

        # get repressors and replicate number
        job_name = job.split('/')[-1]
        replicate = 0
        if '_' in job_name:
            job_name, replicate = job_name.split('_')
        repressor1, repressor2 = job_name.split('-')

        scores_for_each_condition = {}

        # load data
        timeseries_path = os.path.join(job, 'timeseries.json')
        timeseries_dict = load(timeseries_path)

        # plot each condition
        for condition in (('normal', 'diabetic', 'minute')):

            # load time series
            mutant_timeseries_js = timeseries_dict['mutant_'+condition]
            wildtype_timeseries_js = timeseries_dict['wildtype_'+condition]
            mutant_model = GaussianModel.from_json(mutant_timeseries_js)
            wildtype_model = GaussianModel.from_json(wildtype_timeseries_js)

            # convert to deviation form
            focal_point = input_start * 60
            wildtype_deviation_model = wildtype_model.get_deviation_model(wildtype_model.mean[:, focal_point])
            mutant_deviation_model = mutant_model.get_deviation_model(mutant_model.mean[:, focal_point])

            # score data
            commitment_time, commitment_threshold, failure_frequency = get_error_frequency(wildtype_deviation_model, mutant_deviation_model,dim=dim, normalization='deviation')

            scores_for_each_condition[condition] = [failure_frequency]

        scores_for_each_pairing[(repressor1, repressor2)] = scores_for_each_condition

    return scores_for_each_pairing

