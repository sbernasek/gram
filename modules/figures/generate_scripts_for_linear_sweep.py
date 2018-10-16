#!/usr/bin/env python

import os
import sys
import shutil
parentPath = os.path.abspath("../")
if parentPath not in sys.path:
    sys.path.insert(0, parentPath)

import numpy as np
from modules.data_handling import write
import modules.sobol_sampling as sobol_sampling
from time import strftime

model = 'linear'
input_scaling = False
basal_input = 0.

# create directory for current simulation
sim_name = model + '_sweep'
sim_dir_name = '_'.join([sim_name, strftime("%m-%d")])
sim_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'data', sim_dir_name))
i = 0
while os.path.isdir(sim_path):
    i += 1
    sim_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'data', sim_dir_name+'_v'+str(i)))
    if os.path.isdir(sim_path) == True:
            break
os.mkdir(sim_path)

# create directories for bash scripts and simulation results
scripts_path = os.path.join(sim_path, 'scripts')
data_path = os.path.join(sim_path, 'data')
os.mkdir(scripts_path), os.mkdir(data_path)

# copy run script to simulation working directory
shutil.copy(os.path.join(os.getcwd(), 'run_parameters.py'), sim_path)

# write executable file to run all scripts
run_scripts = os.path.join(scripts_path, 'run.sh')
with open(run_scripts, 'w') as wcfile:
    wcfile.write('#!/bin/bash\n\n')
    wcfile.write('for file in qsub\_job\_script*.sh; do qsub $file; done')
    wcfile.close()
os.chmod(run_scripts, 0o755)

# define parameter ranges, log10(val)
base = np.array([0,  # activation
                 0,  # transcription
                 0,  # translation
                 0, # deactivation
                 -2, # mrna degradation
                 -3, # protein degradation
                 -4.5, # eta 1
                 -4.5, # eta 2
                 -4.5, # eta 3
                 ])

delta = 0.55
low, high = base - delta, base + delta

# generate samples
num_samples = 2500
sobol_samples = sobol_sampling.generate_sobol_vectors(dim=len(base), sample_size=num_samples)
sampled_parameters = np.apply_along_axis(sobol_sampling.get_parameter_values, 1, sobol_samples, low=low, high=high)

# write samples to file
write(sampled_parameters.tolist(), os.path.join(sim_path, 'samples.json'))

# create individual qsub script for each set of parameter values
for job_ID, sample in enumerate(sampled_parameters):

    job_name = model + '_%05d' % (job_ID+1)

    # expand list of parameters
    k_activation, k_transcription, k_translation, gamma_d, gamma_r, gamma_p, eta1, eta2, eta3 = sample

    parameters = {
            'baseline': 0.0, # no importance
            'k_activation': k_activation,
            'k_transcription': k_transcription,
            'k_translation': k_translation,

            'gamma_d': gamma_d,
            'gamma_r': gamma_r,
            'gamma_p': gamma_p}

    feedback_parameters = {
            'eta1': eta1,
            'eta2': eta2,
            'eta3': eta3}

    # create directory for job data
    job_data_path = os.path.join(data_path, job_name)
    os.mkdir(job_data_path)

    # add parameters to job data directory
    write(parameters, os.path.join(job_data_path, 'p.json'))
    write(feedback_parameters, os.path.join(job_data_path, 'fb.json'))

    # open the job submission file and name it
    new_file = os.path.join(scripts_path, 'qsub_job_script_%s_phoenix.sh' % job_name)
    wcfile = open(new_file, 'w')

    # define script language
    wcfile.write('#! /bin/bash\n\n')

    # write all Torque parameters
    wcfile.write('#PBS -d %s\n' % sim_path) # directory of execution
    wcfile.write('#PBS -e %s/std.err\n' % (job_data_path))
    wcfile.write('#PBS -o %s/std.out\n' % (job_data_path))
    wcfile.write('#PBS -N %s\n' % job_name)
    wcfile.write('#PBS -q low\n')
    wcfile.write('python3.4 ./run_parameters.py %s %s %s %s\n' % (model, job_data_path, basal_input, int(input_scaling)))

    # close the file
    wcfile.close()

    # change the permissions
    os.chmod(new_file, 0o755)
