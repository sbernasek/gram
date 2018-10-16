#!/usr/bin/env python

import os
import sys
import shutil
parentPath = os.path.abspath("../")
if parentPath not in sys.path:
    sys.path.insert(0, parentPath)

import numpy as np
from itertools import product
from modules.data_handling import write
from parameters import parameters, feedback_parameters
from time import strftime

model = 'linear'

# create directory for current simulation
sim_name = model
sim_dir_name = '_'.join([sim_name, strftime("%m-%d")])
sim_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'data', sim_dir_name))
if os.path.isdir(sim_path):
    raise ValueError('{:s} simulation directory already exists.'.format(sim_dir_name))
os.mkdir(sim_path)

# create directories for bash scripts and simulation results
scripts_path = os.path.join(sim_path, 'scripts')
data_path = os.path.join(sim_path, 'data')
os.mkdir(scripts_path), os.mkdir(data_path)

# copy run script to simulation working directory
shutil.copy(os.path.join(os.getcwd(), 'run_network_under_condition.py'), sim_path)

# write executable file to run all scripts
run_scripts = os.path.join(scripts_path, 'run.sh')
with open(run_scripts, 'w') as wcfile:
    wcfile.write('#!/bin/bash\n\n')
    wcfile.write('for file in qsub\_job\_script*.sh; do qsub $file; done')
    wcfile.close()
os.chmod(run_scripts, 0o755)

def get_simulation_name(targets, condition):
    return '_'.join([''.join(targets), condition])

# enumerate all simulations
conditions = ['normal', 'diabetic', 'minute']
is_within_constraints = lambda x: np.logical_and(sum(x)<=2, sum(x)>0)
repressor_vectors = np.array([i for i in product(*(range(i+1) for i in [2, 2, 2])) if is_within_constraints(i)])

# generate script for each simulation
for repressor_vector in repressor_vectors:
    for condition in conditions:

        # get counts for each repressor
        gene, transcript, protein = repressor_vector

        # define job name
        job_name = get_simulation_name(repressor_vector.astype(str), condition)

        # create directory for job data
        job_data_path = os.path.join(data_path, job_name)
        os.mkdir(job_data_path)

        # add parameters to job data directory
        write(parameters[model], os.path.join(job_data_path, 'p.json'))
        write(feedback_parameters[model], os.path.join(job_data_path, 'fb.json'))

        # open the job submission file and name it
        new_file = os.path.join(scripts_path, 'qsub_job_script_%s_phoenix.sh' % job_name)
        wcfile = open(new_file, 'w')

        # define script language
        wcfile.write('#! /bin/bash\n\n')

        # write all Torque parameters
        wcfile.write('#PBS -d %s\n' % sim_path) # directory of execution
        wcfile.write('#PBS -e %s/std.err\n' % (job_data_path))
        wcfile.write('#PBS -o %s/std.out\n' % (job_data_path))
        wcfile.write('#PBS -N %s\n' % '_'.join([sim_name, job_name]))
        wcfile.write('#PBS -q low\n')
        wcfile.write('python3.4 ./run_network_under_condition.py %s %s %s %s %s %s\n' % (model, gene, transcript, protein, condition, job_data_path))

        # close the file
        wcfile.close()

        # change the permissions
        os.chmod(new_file, 0o755)
