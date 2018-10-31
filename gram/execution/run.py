from os import getcwd
from os.path import join
from time import time
from argparse import ArgumentParser

from gram.simulation.environment import ConditionSimulation


# ======================== PARSE SCRIPT ARGUMENTS =============================

parser = ArgumentParser(description='Run a perturbation simulation.')

# simulation directory
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
parser.add_argument('-D', '--use_deviations',
                    help='Use deviation variables.',
                    type=int,
                    default=False,
                    required=False)

args = vars(parser.parse_args())
simulation_path = args['path'][0]

# ============================= RUN SCRIPT ====================================

# load simulation
simulation = ConditionSimulation.load(simulation_path)

# generate seed for random number generator
seed = int(time())

# run simulation and comparison
simulation.simulate(N=args['trajectories'], seed=seed)
simulation.compare(deviations=bool(args['use_deviations']), inplace=True)

# save simulation
simulation.save(simulation_path, saveall=bool(args['saveall']))