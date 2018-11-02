from os import getcwd
from os.path import join
from argparse import ArgumentParser

from gram.simulation.environment import ConditionSimulation


# ======================== PARSE SCRIPT ARGUMENTS =============================

parser = ArgumentParser(description='Run a batch of simulations.')

# batch file
parser.add_argument('path',
                    nargs=1)

# save simulation trajectories
parser.add_argument('-S', '--saveall',
                    help='Save simulation trajectories.',
                    type=int,
                    default=0,
                    required=False)

# number of trajectories
parser.add_argument('-N', '--trajectories',
                    help='Number of stochastic simulations.',
                    type=int,
                    default=1000,
                    required=False)

# number of trajectories
parser.add_argument('-D', '--deviations',
                    help='Use deviation variables.',
                    type=int,
                    default=False,
                    required=False)

args = vars(parser.parse_args())

# ============================= RUN SCRIPT ====================================

# define simulation arguments
skwargs = dict(N=args['trajectories'])
ckwargs = dict(deviations=bool(args['deviations']))
saveall = bool(args['saveall'])

# run each simulation in batch file
with open(args['path'][0], 'r') as batch_file:

     # run each simulation
     for path in batch_file.readlines():

          # load simulation
          simulation = ConditionSimulation.load(path.strip())

          # run simulation and comparison
          simulation.run(skwargs=skwargs, ckwargs=ckwargs)

          # save simulation
          simulation.save(path.strip(), saveall=saveall)
