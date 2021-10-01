"""
This is the starting file for the Hierarchical Actor-Critc (HAC) algorithm.  The below script processes the command-line options specified
by the user and instantiates the environment and agent. 
"""

import importlib
import os
from options import parse_options
import utils
from run_HAC import run_HAC

# Determine training options specified by user.  The full list of available options can be found in "options.py" file.
FLAGS = parse_options()
utils.set_logger('train.log')
designer = importlib.import_module(f'experiments.{FLAGS.env}.design_agent_and_env')
model_json_path = os.path.join('experiments', FLAGS.env, FLAGS.model, 'params.json')
assert os.path.isfile(model_json_path), f'No model json config file found at {model_json_path}'
params = utils.Params(model_json_path)
print('params: ', params)
params.update(params=FLAGS)

# Instantiate the agent and Mujoco environment. The designer must assign values to the hyperparameters listed in the
# "design_agent_and_env.py" file.
print('FLAGS: ', params)
agent, env = designer.design_agent_and_env(params)

# Begin training
run_HAC(params, env, agent)
