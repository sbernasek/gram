from os.path import join, abspath, relpath, isdir
from os import mkdir, chmod, pardir
import shutil
import pickle
import numpy as np
from time import time
from datetime import datetime

from .sampling import LogSampler
from ..simulation.environment import ConditionSimulation
from ..models.linear import LinearModel
from ..models.hill import HillModel
from ..models.twostate import TwoStateModel


class Sweep:
    """
    Class defines a single parameter sweep of a given model.

    Attributes:

        path (str) - path to sweep directory

        simulation_paths (dict) - paths to simulation directories

        base (np.ndarray[float]) - base parameter values

        delta (float or np.ndarray[float]) - log-deviations about base

        sampler (LogSampler) - sobol sample generator

        parameters (np.ndarray[float]) - sampled parameter values

        sim_kw (dict) - keyword arguments for simulation

    """

    def __init__(self, base, delta=0.5):
        """
        Instantiate parameter sweep.

        Args:

            base (np.ndarray[float]) - base parameter values

            delta (float or np.ndarray[float]) - log-deviations about base

        """

        self.base = base
        self.delta = delta
        self.sampler = LogSampler(base-delta, base+delta)

    @property
    def N(self):
        """ Number of samples. """
        return self.samples.shape[0]

    @staticmethod
    def load(path):
        """ Load sweep from target <path>. """
        with open(join(path, 'sweep.pkl'), 'rb') as file:
            sweep = pickle.load(file)
        return sweep

    @staticmethod
    def build_submission_script(path,
                                num_trajectories,
                                saveall,
                                use_deviations,
                                allocation='p30653'):
        """
        Writes job submission script.

        Args:

            path (str) - sweep path

            num_trajectories (int) - number of simulation trajectories

            saveall (bool) - if True, save simulation trajectories

            use_deviations (bool) - if True, use deviation variables

            allocation (str) - project allocation, e.g. p30653 (comp. bio)

        """

        # define paths
        sweep_path = abspath(path)
        job_path = join(sweep_path, 'scripts', 'job_submission.sh')

        # copy run script to scripts directory
        run_path = join(abspath(__file__).rsplit('/', maxsplit=1)[0], 'run.py')
        shutil.copy(run_path, join(self.path, 'scripts'))

        # declare outer script that reads PATH from file
        job_script = open(job_path, 'w')
        job_script.write('#!/bin/bash\n')
        job_script.write('while IFS=$\'\\t\' read PATH\n')
        job_script.write('do\n')
        job_script.write('\tJOB=`msub - << EOJ\n\n')

        # =========== begin submission script for individual job =============
        job_script.write('#! /bin/bash\n')
        job_script.write('#MSUB -A {:s} \n'.format(allocation))
        job_script.write('#MSUB -q short \n')
        job_script.write('#MSUB -l walltime=04:00:00 \n')
        job_script.write('#MSUB -m abe \n')
        job_script.write('#MSUB -M sebastian@u.northwestern.edu \n')
        job_script.write('#MSUB -j oe \n')
        #job_script.write('#MSUB -N %s \n' % job_id)
        job_script.write('#MSUB -l nodes=1:ppn=2 \n')
        job_script.write('#MSUB -l mem=1gb \n\n')

        # load python module
        job_script.write('module load python/anaconda3.6\n\n')

        # set working directory and run script
        job_script.write('cd {:s} \n\n'.format(sweep_path))

        # run script
        args = (num_trajectories, saveall, use_deviations)
        job_script.write('python scripts/run.py \
                         $\{PATH} -N {:d} -S {:s} -D {:s}\n'.format(*args))
        job_script.write('EOJ\n')
        job_script.write('`\n\n')
        # ============= end submission script for individual job =============

        # print job id
        job_script.write('echo "JobID = $\{JOB} submitted on `date`"\n')
        job_script.write('done < /scripts/paths.txt \n')
        job_script.write('exit\n')

        # close the file
        job_script.close()

        # change the permissions
        chmod(job_path, 0o755)

    def build_paths_file(self):
        """ Writes file containing all simulation paths. """
        paths = open(join(self.path, 'scripts', 'paths.txt'), 'w')
        for path in self.simulation_paths.values():
            paths.write('{:s} \n'.format(path))
        paths.close()

    def build_sweep_directory(self, directory='./'):
        """
        Create directory for sweep.

        Args:

            directory (str) - destination path

        """

        # assign name to sweep
        timestamp = datetime.fromtimestamp(time()).strftime('%y%m%d_%H%M%S')
        name = '{:s}_{:s}'.format(self.__class__.__name__, timestamp)

        # create directory (overwrite existing one)
        path = join(directory, name)
        if not isdir(path):
            mkdir(path)
        self.path = path

        # make subdirectories for simulations and scripts
        mkdir(join(path, 'scripts'))
        mkdir(join(path, 'simulations'))

    def build(self,
              directory='./',
              num_samples=10,
              num_trajectories=1000,
              saveall=False,
              use_deviations=False,
              allocation='p30653',
              **sim_kw):
        """
        Build directory tree for a parameter sweep. Instantiates and saves a simulation instance for each parameter sample, then generates a single shell script to submit each simulation as a separate batch job.

        Args:

            directory (str) - destination path

            num_samples (int) - number of samples in parameter space

            num_trajectories (int) - number of simulation trajectories

            saveall (bool) - if True, save simulation trajectories

            use_deviations (bool) - if True, use deviation variables

            allocation (str) - project allocation

            sim_kw (dict) - keyword arguments for ConditionSimulation

        """

        # create sweep directory
        self.build_sweep_directory(directory)

        # store parameters
        self.parameters = self.sampler.sample(num_samples)
        self.sim_kw = sim_kw
        self.simulation_paths = {}

        # build simulations
        for i, parameters in enumerate(self.parameters):
            simulation_path = join(self.path, 'simulations', '{:d}'.format(i))
            self.simulation_paths[i] = simulation_path
            self.build_simulation(parameters, simulation_path, **sim_kw)

        # save serialized sweep
        with open(join(self.path, 'sweep.pkl'), 'wb') as file:
            pickle.dump(self, file, protocol=-1)

        # build parameter file
        self.build_paths_file()

        # build job submission script
        self.build_submission_script(self.path,
                                     num_trajectories,
                                     saveall,
                                     use_deviations,
                                     allocation=allocation)

    @staticmethod
    def build_model(parameters):
        """
        Returns a model instance defined by the provided parameters.

        Args:

            parameters (np.ndarray[float]) - model parameters

        Returns:

            model (Cell instance)

        """
        pass

    @classmethod
    def build_simulation(cls, parameters, simulation_path, **kwargs):
        """
        Builds and saves a simulation instance for a set of parameters.

        Args:

            parameters (np.ndarray[float]) - parameter values

            simulation_path (str) - simulation path

            kwargs: keyword arguments for ConditionSimulation

        """

        # build model
        model = cls.build_model(parameters)

        # instantiate simulation
        simulation = ConditionSimulation(model, **kwargs)

        # create simulation directory
        if not isdir(simulation_path):
            mkdir(simulation_path)

        # save simulation
        simulation.save(simulation_path)


class LinearSweep(Sweep):

    """
    Parameter sweep for linear model. Parameters are:

        0: activation rate constant
        1: transcription rate constant
        2: translation rate constant
        3: deactivation rate constant
        4: mrna degradation rate constant
        5: protein degradation rate constant
        6: transcriptional feedback strength
        7: post-transcriptional feedback strength
        8: post-translational feedback strength

    """

    def __init__(self, base=None, delta=0.5):
        """
        Instantiate parameter sweep.

        Args:

            base (np.ndarray[float]) - base parameter values

            delta (float or np.ndarray[float]) - log-deviations about base

        """

        # define parameter ranges, log10(val)
        if base is None:
            base = np.array([0, 0, 0, 0, -2, -3, -4.5, -4.5, -4.5])

        # call parent instantiation
        super().__init__(base, delta)

    @staticmethod
    def build_model(parameters):
        """
        Returns a model instance defined by the provided parameters.

        Args:

            parameters (np.ndarray[float]) - model parameters

        Returns:

            model (LinearModel)

        """

        # extract parameters
        k0, k1, k2, g0, g1, g2, eta0, eta1, eta2 = parameters

        # instantiate base model
        model = LinearModel(k0=k0, k1=k1, k2=k2, g0=g0, g1=g1, g2=g2)

        # add feedback (two equivalent sets)
        model.add_feedback(eta0, eta1, eta2, perturbed=False)
        model.add_feedback(eta0, eta1, eta2, perturbed=True)

        return model


class HillSweep(Sweep):

    """
    Parameter sweep for hill model. Parameters are:

        0: transcription hill coefficient
        1: transcription rate constant
        2: translation rate constant
        3: mrna degradation rate constant
        4: protein degradation rate constant
        5: repressor michaelis constant
        6: repressor hill coefficient
        7: post-transcriptional feedback strength
        8: post-translational feedback strength

    """

    def __init__(self, base=None, delta=0.5):
        """
        Instantiate parameter sweep.

        Args:

            base (np.ndarray[float]) - base parameter values

            delta (float or np.ndarray[float]) - log-deviations about base

        """

        # define parameter ranges, log10(val)
        if base is None:
            base = np.array([0, 0, 0, -2, -3, -4, 0, -5, -4])

        # call parent instantiation
        super().__init__(base, delta)

    @staticmethod
    def build_model(parameters):
        """
        Returns a model instance defined by the provided parameters.

        Args:

            parameters (np.ndarray[float]) - model parameters

        Returns:

            model (HillModel)

        """

        # extract parameters
        n, k1, k2, g1, g2, k_m, r_n, eta1, eta2 = parameters

        # instantiate base model
        model = HillModel(k1=k1, k_m=.5, n=n, k2=k2, g1=g1, g2=g2)

        # add feedback (two equivalent sets)
        model.add_feedback(k_m, r_n, eta1, eta2, perturbed=False)
        model.add_feedback(k_m, r_n, eta1, eta2, perturbed=True)

        return model

class TwoStateSweep(Sweep):

    """
    Parameter sweep for twostate model. Parameters are:

        0: activation rate constant
        1: transcription rate constant
        2: translation rate constant
        3: deactivation rate constant
        4: mrna degradation rate constant
        5: protein degradation rate constant
        6: transcriptional feedback strength
        7: post-transcriptional feedback strength
        8: post-translational feedback strength

    """

    def __init__(self, base=None, delta=0.5):
        """
        Instantiate parameter sweep.

        Args:

            base (np.ndarray[float]) - base parameter values

            delta (float or np.ndarray[float]) - log-deviations about base

        """

        # define parameter ranges, log10(val)
        if base is None:
            base = np.array([0, 0, 0, -1, -2, -3, -4, -4.5, -4])

        # call parent instantiation
        super().__init__(base, delta)

    @staticmethod
    def build_model(parameters):
        """
        Returns a model instance defined by the provided parameters.

        Args:

            parameters (np.ndarray[float]) - model parameters

        Returns:

            model (LinearModel)

        """

        # extract parameters
        k0, k1, k2, g0, g1, g2, eta0, eta1, eta2 = parameters

        # instantiate base model
        model = TwoStateModel(k0=k0, k1=k1, k2=k2, g0=g0, g1=g1, g2=g2)

        # add feedback (two equivalent sets)
        model.add_feedback(eta0, eta1, eta2, perturbed=False)
        model.add_feedback(eta0, eta1, eta2, perturbed=True)

        return model
