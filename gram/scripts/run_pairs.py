import numpy as np
import pickle
from gram.models.linear import LinearModel
from gram.simulation.environment import ConditionSimulation


N = 10

# define feedback strengths
eta = (1e-4, 1e-4, 1e-4)

# run pairwise simulations
simulations = {}
for i in range(3):
    for j in range(3):

        # define feedback strengths
        permanent = np.zeros(3)
        permanent[i] = eta[i]
        removed = np.zeros(3)
        removed[j] = eta[j]

        # define model
        model = LinearModel(g1=0.01, g2=0.001)
        model.add_feedback(*permanent)
        model.add_feedback(*removed, perturbed=True)

        # run simulation
        sim = ConditionSimulation(model)
        sim.run(skwargs=dict(N=N))
        simulations[(i, j)] = sim

# save simulations
with open('pairwise_sweep.pkl', 'wb') as file:
    pickle.dump(simulations, file)
