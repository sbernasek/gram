# navigate to modules folder
import os
import sys
parentPath = os.path.abspath("../../")
if parentPath not in sys.path:
    sys.path.insert(0, parentPath)

import time
import copy
import numpy as np
from modules.feedback import FeedbackSystem
from modules.analysis import evaluate_error_frequency as evaluate
from modules.data_handling import write as write_json

basal_input = 0.0
num_trials = 2500
dt = 1
input_start = 0
duration = 100
method = 'cy_hybrid'
normalization = None
save_whole_time_series = False

# get parameters
script, model, target1, target2, data_path = sys.argv
p_path = os.path.join(data_path, 'p.json')
fb_path = os.path.join(data_path, 'fb.json')

# start clock
start = time.time()

# parse repressor targets
mut_targets = {'gene': 0, 'transcript': 0, 'protein': 0}
mut_targets[target1] += 1
wt_targets = copy.deepcopy(mut_targets)
wt_targets[target2] += 1

# define cells
wildtype = FeedbackSystem(model=model, gene=wt_targets['gene'], transcript=wt_targets['transcript'],
                          protein=wt_targets['protein'], params=p_path, feedback_params=fb_path)
mutant = FeedbackSystem(model=model, gene=mut_targets['gene'], transcript=mut_targets['transcript'],
                        protein=mut_targets['protein'], params=p_path, feedback_params=fb_path)

# initialize results dictionaries
error_frequencies = {}
time_series = {}

# if model is twostate, initialize diploid cell
if model == 'twostate':
    ic = np.zeros(len(wildtype.nodes), dtype=np.int64)
    ic[0] = 2
else:
    ic = None

# run simulation
for condition in ('normal', 'diabetic', 'minute'):
    error_frequency, wt_model, mut_model = evaluate(wildtype, mutant, condition, ic=ic, basal_input=basal_input, num_trials=num_trials, dt=dt, duration=duration, input_start=input_start, normalization=normalization, method=method)

    # store error frequency
    error_frequencies[condition] = error_frequency.tolist()

    # store time_series
    time_series['wildtype_' + condition] = wt_model.to_json(retall=save_whole_time_series)
    time_series['mutant_' + condition] = mut_model.to_json(retall=save_whole_time_series)

# write results to file
write_json(error_frequencies, os.path.join(data_path, 'error_frequencies.json'))
write_json(wildtype.to_json(), os.path.join(data_path, 'wildtype.json'))
write_json(mutant.to_json(), os.path.join(data_path, 'mutant.json'))
write_json(time_series, os.path.join(data_path, 'timeseries.json'))

stop = time.time()

print('Model used:', model)
print('Repressor 1:', target1)
print('Repressor 2:', target2)
print('Algorithm:', method)
print('Normalization', normalization)
print('{:d} trials took {:0.2f} seconds.\n'.format(num_trials, stop-start))

print('\n \n')
wildtype.print_reactions()
