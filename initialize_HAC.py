"""
This is the starting file for the Hierarchical Actor-Critc (HAC) algorithm.  The below script processes the command-line options specified
by the user and instantiates the environment and agent. 
"""

import importlib
from options import parse_options
import utils
from run_HAC import run_HAC

# Determine training options specified by user.  The full list of available options can be found in "options.py" file.
FLAGS = parse_options()
utils.set_logger('train.log')
designer = importlib.import_module(f'experiments.{FLAGS.env}.design_agent_and_env')

# Instantiate the agent and Mujoco environment. The designer must assign values to the hyperparameters listed in the
# "design_agent_and_env.py" file.
print('FLAGS: ', FLAGS)
agent, env = designer.design_agent_and_env(FLAGS)

# Begin training
run_HAC(FLAGS, env, agent)
