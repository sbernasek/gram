from gram.execution.arguments import SweepArguments
from gram.sweep.sweep import LinearSweep, HillSweep, TwoStateSweep


# ======================== PARSE SCRIPT ARGUMENTS =============================

args = SweepArguments(description='Parameter sweep arguments.')
model = args['model']
num_samples = args['number_of_samples']

# ============================= RUN SCRIPT ====================================

# instantiate sweep object
if model == 'linear':
    sweep = LinearSweep(num_samples=num_samples)
elif model == 'hill':
    sweep = HillSweep(num_samples=num_samples)
elif model == 'twostate':
    sweep = TwoStateSweep(num_samples=num_samples)
else:
    raise ValueError('{:s} model type not recognized.'.format(model))

# build sweep
sweep.build(
    directory=args['path'],
    batch_size=args['batch_size'],
    num_trajectories=args['number_of_trajectories'],
    saveall=args['save_all'],
    horizon=args['horizon'],
    deviations=args['use_deviations'],
    walltime=args['walltime'],
    allocation=args['allocation'],
    pulse_start=args['pulse_start'],
    pulse_duration=args['pulse_duration'],
    pulse_baseline=args['pulse_baseline'],
    pulse_magnitude=args['pulse_magnitude'],
    pulse_sensitive=args['pulse_sensitive'],
    simulation_duration=args['simulation_duration'])

"""

# basal input
python build_sweep.py -N 5000 -n 2500 -b 25 -s 0 -w 10 -pb 0.1 -ps 50 -sd 250 -d true

"""
