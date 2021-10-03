"""
This is the starting file for the Hierarchical Actor-Critc (HAC) algorithm.  The below script processes the command-line options specified
by the user and instantiates the environment and agent. 
"""

import importlib
import os
from options import parse_options
import utils
from run_HAC import run_HAC
from run_HAC_lvl_parallel import run_HAC_lvl_parallel

# Determine training options specified by user.  The full list of available options can be found in "options.py" file.
FLAGS = parse_options()
designer = importlib.import_module(f'experiments.{FLAGS.env}.design_agent_and_env')
model_json_path = os.path.join('experiments', FLAGS.env, FLAGS.model, 'params.json')
assert os.path.isfile(model_json_path), f'No model json config file found at {model_json_path}'
params = utils.Params(model_json_path)
params.update(params=FLAGS)
if FLAGS.exp_name is not None:
    params.model_dir = os.path.join('experiments', FLAGS.env, FLAGS.model, FLAGS.exp_name)
else:
    params.model_dir = os.path.join('experiments', FLAGS.env, FLAGS.model)
if not os.path.exists(params.model_dir):
    os.makedirs(params.model_dir)
utils.set_logger(os.path.join(params.model_dir, 'train.log'))

# Instantiate the agent and Mujoco environment. The designer must assign values to the hyperparameters listed in the
# "design_agent_and_env.py" file.
agent, env = designer.design_agent_and_env(params)

# Begin training
if params.lvl_parallel:
    run_HAC_lvl_parallel(params, env, agent)
else:
    run_HAC(params, env, agent)
