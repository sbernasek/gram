# navigate to modules folder
import os
import sys
parentPath = os.path.abspath("../../")
if parentPath not in sys.path:
    sys.path.insert(0, parentPath)

import time
import numpy as np
from modules.feedback import FeedbackSystem
from modules.data_handling import write as write_json
from nevosim.solver.signals import cSquarePulse
time_scaling = 1/60

method = 'cy_hybrid'
num_trials = 2500
dt = 1

basal_input = 0.
input_start = 0
input_duration = 3
input_scaling = False
duration = 100 / time_scaling

normalization = None
save_trajectories = False

# get parameters
script, model, gene, transcript, protein, condition, data_path = sys.argv
p_path = os.path.join(data_path, 'p.json')
fb_path = os.path.join(data_path, 'fb.json')

# define system
network = FeedbackSystem(model=model, gene=int(gene), transcript=int(transcript), protein=int(protein), params=p_path, feedback_params=fb_path)

# if model is twostate, initialize diploid cell
if model == 'twostate':
    ic = np.zeros(len(network.nodes), dtype=np.int64)
    ic[0] = 2
else:
    ic = None

# # define input signal
# input_signal = lambda t: 1 if t < (input_start+input_duration)/time_scaling and t > input_start/time_scaling else basal_input

# instantiate cythonized input signal
t_on, t_off = input_start/time_scaling, (input_start+input_duration)/time_scaling
input_signal = cSquarePulse(t_on, t_off, off=basal_input, on=1)

# run monte carlo simulations
start = time.time()
trajectories, time_series_model = network.run_stochastic_simulation(input_signal, ic=ic, condition=condition, normalization=normalization, num_trials=num_trials, dt=dt, duration=duration, method=method)
stop = time.time()

# write results as json
write_json(network.to_json(), os.path.join(data_path, 'network.json'))
time_series_json = time_series_model.to_json(retall=save_trajectories)
write_json(time_series_json, os.path.join(data_path, 'timeseries.json'))

# print summary to standard output file
print('Model: ', model)
print('Gene Regulators: ', gene)
print('Transcript Regulators: ', transcript)
print('Protein Regulators: ', protein)
print('Condition: ', condition)
print('Basal Input: ', basal_input)
print('Algorithm: ', method)
print('Num. Trials: ', num_trials)
print('Normalization: ', normalization)
print('Input scaling:', input_scaling)
print('{:d} trials took {:0.2f} seconds.\n'.format(num_trials, stop-start))
print('\n \n')
network.print_reactions()
