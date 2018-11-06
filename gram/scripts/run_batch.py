from time import time
from gram.simulation.environment import ConditionSimulation
from gram.execution.arguments import RunArguments


# ======================== PARSE SCRIPT ARGUMENTS =============================

args = RunArguments(description='Simulation arguments.')
skwargs = dict(N=args['number_of_trajectories'], debug=args['debug'])
ckwargs = dict(horizon=args['horizon'], deviations=args['use_deviations'])


# ============================= RUN SCRIPT ====================================

start_time = time()

# run each simulation in batch file
with open(args['path'], 'r') as batch_file:

     # run each simulation
     for path in batch_file.readlines():

          # load simulation
          simulation = ConditionSimulation.load(path.strip())

          # run simulation and comparison
          simulation.run(skwargs=skwargs, ckwargs=ckwargs)

          # save simulation
          simulation.save(path.strip(), saveall=args['save_all'])

# print runtime to standard out
runtime = time() - start_time
print('\nSIMULATION COMPLETED IN {:0.2f}.\n'.format(runtime))
