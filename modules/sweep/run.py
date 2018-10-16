import sys
from modules.simulation.environment import ConditionSimulation

# get simulation path
path = sys.argv[1]

# load simulation
simulation = ConditionSimulation.load(path)

# run simulation and comparison
simulation.simulate(N=500)
simulation.compare(deviations=False, inplace=True)

# save simulation
simulation.save(path, saveall=True)
