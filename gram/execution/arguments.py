from os import getcwd
from argparse import ArgumentParser, ArgumentTypeError


# ======================== ARGUMENT TYPE CASTING ==============================

def str2bool(arg):
     """ Convert <arg> to boolean. """
     if arg.lower() in ('yes', 'true', 't', 'y', '1'):
          return True
     elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
          return False
     else:
          raise ArgumentTypeError('Boolean value expected.')


# ======================== PARSE SCRIPT ARGUMENTS =============================


class RunArguments(ArgumentParser):
     """ Argument handler for run scripts. """

     def __init__(self, **kwargs):
          super().__init__(**kwargs)
          self.add_arguments()
          self.parse()

     def __getitem__(self, key):
          """ Returns <key> argument value. """
          return self.args[key]

     def add_arguments(self):
          """ Add arguments. """

          # add position argument for path
          self.add_argument(
               'path',
               nargs='?',
               default=getcwd())

          # add keyword argument for number of simulated trajectories
          self.add_argument(
               '-N', '--number_of_trajectories',
               help='Number of stochastic simulation trajectories.',
               type=int,
               default=1000,
               required=False)

          # add keyword argument for using deviation variables
          self.add_argument(
               '-D', '--use_deviations',
               help='Use deviation variables.',
               type=str2bool,
               default=False,
               required=False)

          # add keyword argument for saving simulated trajectories
          self.add_argument(
               '-S', '--save_all',
               help='Save simulation trajectories.',
               type=str2bool,
               default=False,
               required=False)

     def parse(self):
          """ Parse arguments. """
          self.args = vars(self.parse_args())


class SweepArguments(RunArguments):
     """ Argument handler for parameter sweeps. """

     def add_arguments(self):
          """ Add arguments. """

          super().add_arguments()

          # add keyword argument for model used
          self.add_argument('-m', '--model',
                              help='Model type.',
                              type=str,
                              default='linear',
                              required=False)

          # add keyword argument for number of parameter samples
          self.add_argument('-n', '--number_of_samples',
                              help='Number of parameter samples.',
                              type=int,
                              default=10,
                              required=False)

          # add keyword argument for batch size
          self.add_argument('-b', '--batch_size',
                              help='Number of simulations per batch.',
                              type=int,
                              default=25,
                              required=False)

          # add keyword argument for project allocation
          self.add_argument('-w', '--walltime',
                              help='Estimated run time.',
                              type=int,
                              default=10,
                              required=False)

          # add keyword argument for project allocation
          self.add_argument('-A', '--allocation',
                              help='Project allocation.',
                              type=str,
                              default='p30653',
                              required=False)

