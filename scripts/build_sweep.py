from os import getcwd
from argparse import ArgumentParser

from gram.sweep import sweep


# ======================== PARSE SCRIPT ARGUMENTS =============================

parser = ArgumentParser(description='Generate a parameter sweep.')

# sweeps directory
parser.add_argument('path',
                    nargs='?',
                    default=getcwd())

# model type
parser.add_argument('-m', '--model',
                    help='Model type.',
                    type=str,
                    default='linear',
                    required=False)

# number of parameter samples
parser.add_argument('-n', '--num_samples',
                    help='Number of parameter samples.',
                    type=int,
                    default=10,
                    required=False)

# number of stochastic simulation trajectories
parser.add_argument('-N', '--num_trajectories',
                    help='Number of stochastic simulation trajectories.',
                    type=int,
                    default=1000,
                    required=False)

# save simulation trajectories
parser.add_argument('-S', '--saveall',
                    help='Save simulation trajectories.',
                    type=bool,
                    default=False,
                    required=False)

# number of trajectories
parser.add_argument('-D', '--use_deviations',
                    help='Use deviation variables.',
                    type=bool,
                    default=False,
                    required=False)

# project allocation
parser.add_argument('-A', '--allocation',
                    help='Project allocation.',
                    type=str,
                    default='p30653',
                    required=False)

args = vars(parser.parse_args())

# ============================= RUN SCRIPT ====================================


# instantiate sweep object
if args['model'] == 'linear':
    sweep_obj = sweep.LinearSweep(num_samples=args['num_samples'])
elif args['model'] == 'hill':
    sweep_obj = sweep.HillSweep(num_samples=args['num_samples'])
elif args['model'] == 'twostate':
    sweep_obj = sweep.TwoStateSweep(num_samples=args['num_samples'])
else:
    raise ValueError('{:s} model type not recognized.'.format(args['model']))

# build sweep
sweep_obj.build(
    directory=args['path'],
    num_trajectories=args['num_trajectories'],
    saveall=args['saveall'],
    use_deviations=args['use_deviations'],
    allocation=args['allocation'])

# USEFUL COMMANDS
# showq -w acct=p30653
