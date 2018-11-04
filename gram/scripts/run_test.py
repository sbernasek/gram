from time import time
from gram.simulation.pulse import PulseSimulation
from gram.simulation.environment import ConditionSimulation
from gram.execution.arguments import PulseArguments


# ======================== PARSE SCRIPT ARGUMENTS =============================

args = PulseArguments(description='Pulse simulation arguments.')

simulation_kwargs = dict(
    pulse_start=args['pulse_start'],
    pulse_duration=args['pulse_duration'],
    pulse_baseline=args['pulse_baseline'],
    pulse_magnitude=args['pulse_magnitude'],
    pulse_sensitive=args['pulse_sensitive'],
    simulation_duration=args['simulation_duration'])


# ============================= RUN SCRIPT ====================================

seed = 0

start_time = time()

# load cell from condition simulation
cell = ConditionSimulation.load(args['path']).cell

# instantiate pulse simulation
simulation = PulseSimulation(cell, **simulation_kwargs)

# run simulation
simulation.run(condition='normal',
               N=args['number_of_trajectories'],
               seed=seed)

# print runtime to standard out
runtime = time() - start_time
print('\nSIMULATION COMPLETED IN {:0.2f}.\n'.format(runtime))
