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
    deviations=args['use_deviations'],
    allocation=args['allocation'])
