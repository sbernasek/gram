from time import time
from gram.simulation.environment import ConditionSimulation
from gram.execution.arguments import RunArguments


# ======================== PARSE SCRIPT ARGUMENTS =============================

args = RunArguments(description='Simulation arguments.')
skwargs = dict(N=args['number_of_trajectories'], debug=args['debug'])
ckwargs = dict(horizon=args['horizon'],
               deviations=args['use_deviations'],
               mode=args['comparison_mode'])
path = args['path']

# ============================= RUN SCRIPT ====================================

start_time = time()

# load simulation
simulation = ConditionSimulation.load(path)

# run simulation and comparison
simulation.run(skwargs=skwargs, ckwargs=ckwargs)

# save simulation
simulation.save(path, saveall=args['save_all'])

# print runtime to standard out
runtime = time() - start_time
print('\nSIMULATION COMPLETED IN {:0.2f}.\n'.format(runtime))
