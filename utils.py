import json
import logging
import os
import shutil
import sys
import numpy as np
from tqdm import tqdm
import torch


logger = logging.getLogger('HAC.utils')


class Params:
    """
    Class that loads hyperparameters from a json file as a dictionary (also support nested dicts).
    Example:
    params = Params(json_path)
    # access key-value pairs
    params.learning_rate
    params['learning_rate']
    # change the value of learning_rate in params
    params.learning_rate = 0.5
    params['learning_rate'] = 0.5
    # print params
    print(params)
    # combine two json files
    params.update(Params(json_path2))
    """

    def __init__(self, json_path=None):
        if json_path is not None and os.path.isfile(json_path):
            with open(json_path) as f:
                params = json.load(f)
                self.__dict__.update(params)
        else:
            self.__dict__ = {}

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4, ensure_ascii=False)

    def update(self, json_path=None, params=None):
        """Loads parameters from json file"""
        if json_path is not None:
            with open(json_path) as f:
                params = json.load(f)
                self.__dict__.update(params)
        elif params is not None:
            self.__dict__.update(vars(params))
        else:
            raise Exception('One of json_path and params must be provided in Params.update()!')

    def __contains__(self, item):
        return item in self.__dict__

    def __getitem__(self, key):
        return getattr(self, str(key))

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __str__(self):
        return json.dumps(self.__dict__, sort_keys=True, indent=4, ensure_ascii=False)


def set_logger(log_path):
    """
    Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    logging.info('Starting training...')
    Args:
        log_path: (string) where to log
    """
    _logger = logging.getLogger('HAC')
    _logger.setLevel(logging.INFO)

    fmt = logging.Formatter('[%(asctime)s] %(name)s: %(message)s', '%m/%d %H:%M:%S')

    class TqdmHandler(logging.StreamHandler):
        def __init__(self, formatter):
            logging.StreamHandler.__init__(self)
            self.setFormatter(formatter)
            self.setStream(tqdm)

        def emit(self, record):
            msg = self.format(record)
            tqdm.write(msg)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    _logger.addHandler(file_handler)
    _logger.addHandler(TqdmHandler(fmt))
    # handler = logging.StreamHandler(stream=sys.stdout)
    # _logger.addHandler(handler)

    # https://stackoverflow.com/questions/6234405/logging-uncaught-exceptions-in-python?noredirect=1&lq=1
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            _logger.info('=*=*=*= Keyboard interrupt =*=*=*=')
            return

        _logger.error("Exception --->", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception


# Below function prints out options and environment specified by user
def print_summary(FLAGS, env):
    print("\n- - - - - - - - - - -")
    print("Task Summary: ", "\n")
    print("Environment: ", env.name)
    print("Number of Layers: ", FLAGS.layers)
    print("Time Limit per Layer: ", FLAGS.time_scale)
    print("Max Episode Time Steps: ", env.max_actions)
    print("Retrain: ", FLAGS.retrain)
    print("Test: ", FLAGS.test)
    print("Visualize: ", FLAGS.show)
    print("- - - - - - - - - - -", "\n\n")


# Below function ensures environment configurations were properly entered
def check_validity(model_name, goal_space_train, goal_space_test, end_goal_thresholds, initial_state_space,
                   subgoal_bounds, subgoal_thresholds, max_actions, timesteps_per_action):
    # Ensure model file is an ".xml" file
    assert model_name[-4:] == ".xml", "Mujoco model must be an \".xml\" file"

    # Ensure upper bounds of range is >= lower bound of range
    if goal_space_train is not None:
        for i in range(len(goal_space_train)):
            assert goal_space_train[i][1] >= goal_space_train[i][
                0], "In the training goal space, upper bound must be >= lower bound"

    if goal_space_test is not None:
        for i in range(len(goal_space_test)):
            assert goal_space_test[i][1] >= goal_space_test[i][
                0], "In the training goal space, upper bound must be >= lower bound"

    for i in range(len(initial_state_space)):
        assert initial_state_space[i][1] >= initial_state_space[i][
            0], "In initial state space, upper bound must be >= lower bound"

    for i in range(len(subgoal_bounds)):
        assert subgoal_bounds[i][1] >= subgoal_bounds[i][0], "In subgoal space, upper bound must be >= lower bound"

        # Make sure end goal spaces and thresholds have same first dimension
    if goal_space_train is not None and goal_space_test is not None:
        assert len(goal_space_train) == len(goal_space_test) == len(
            end_goal_thresholds), "End goal space and thresholds must have same first dimension"

    # Makde sure suboal spaces and thresholds have same dimensions
    assert len(subgoal_bounds) == len(subgoal_thresholds), "Subgoal space and thresholds must have same first dimension"

    # Ensure max action and timesteps_per_action are postive integers
    assert max_actions > 0, "Max actions should be a positive integer"

    assert timesteps_per_action > 0, "Timesteps per action should be a positive integer"


def save_checkpoint(agent, batch, success_rate, file_dir):
    """
    Saves model and training parameters at file_dir + 'last.pth.tar'. If is_best==True, also saves
    file_dir + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as episode, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        file_dir: (string) folder where parameters are to be saved
        loss: (np.array)
        ins_name: (int) instance index
    """
    filepath = os.path.join(file_dir, f'batch_{batch}.pth.tar')
    states = {'num_layers': agent.num_layers,
              'success_rate': success_rate,
              'episode': batch}
    for i in range(agent.num_layers):
        states[f'level_{i}'] = {
            'actor': agent.layers[i].actor.state_dict(),
            'actor_optim': agent.layers[i].actor_optimizer.state_dict(),
            'actor_target': agent.layers[i].actor_target.state_dict(),
            'critic': agent.layers[i].critic.state_dict(),
            'critic_optim': agent.layers[i].critic_optimizer.state_dict(),
            'critic_target': agent.layers[i].critic_target.state_dict(),
        }
    torch.save(states, filepath)
    logger.info(f'file_dir saved to {filepath}')
    shutil.copyfile(filepath, os.path.join(file_dir, 'last.pth.tar'))


def load_checkpoint(agent, file_dir, restore_file):
    """
    Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in file_dir.
    Args:
        restore_file: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from file_dir
    """
    filepath = os.path.join(file_dir, restore_file + '.pth.tar')
    if not os.path.exists(file_dir):
        raise FileNotFoundError(f"File doesn't exist {filepath}")
    else:
        logger.info(f'Restoring parameters from {filepath}')
    if torch.cuda.is_available():
        filepath = torch.load(filepath, map_location='cuda')
    else:
        filepath = torch.load(filepath, map_location='cpu')
    num_layers = filepath['num_layers']
    assert num_layers == agent.num_layers
    for i in range(num_layers):
        agent.layers[i].actor.load_state_dict(filepath[f'level_{i}']['actor'])
        agent.layers[i].actor_optimizer.load_state_dict(filepath[f'level_{i}']['actor_optim'])
        agent.layers[i].actor_target.load_state_dict(filepath[f'level_{i}']['actor_target'])
        agent.layers[i].critic.load_state_dict(filepath[f'level_{i}']['critic'])
        agent.layers[i].critic_optimizer.load_state_dict(filepath[f'level_{i}']['critic_optim'])
        agent.layers[i].critic_target.load_state_dict(filepath[f'level_{i}']['critic_target'])

    logger.info(f"Restored parameters have success rate: {filepath['success_rate']}")
    return filepath['episode']


def load_checkpoint_lvl_parallel(agent, file_dir, levels_batch):
    """
    Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in file_dir.
    Args:
        restore_file: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from file_dir
    """
    assert len(levels_batch) == agent.num_layers
    for i in range(len(levels_batch)):
        filepath = os.path.join(file_dir, f'batch_{levels_batch[i]}.pth.tar')
        if not os.path.exists(file_dir):
            raise FileNotFoundError(f"File doesn't exist {filepath}")
        else:
            logger.info(f'Restoring parameters from {filepath}')
        if torch.cuda.is_available():
            filepath = torch.load(filepath, map_location='cuda')
        else:
            filepath = torch.load(filepath, map_location='cpu')
        num_layers = filepath['num_layers']
        assert num_layers == agent.num_layers
        agent.layers[i].actor.load_state_dict(filepath[f'level_{i}']['actor'])
        agent.layers[i].actor_optimizer.load_state_dict(filepath[f'level_{i}']['actor_optim'])
        agent.layers[i].actor_target.load_state_dict(filepath[f'level_{i}']['actor_target'])
        agent.layers[i].critic.load_state_dict(filepath[f'level_{i}']['critic'])
        agent.layers[i].critic_optimizer.load_state_dict(filepath[f'level_{i}']['critic_optim'])
        agent.layers[i].critic_target.load_state_dict(filepath[f'level_{i}']['critic_target'])

        logger.info(f"Layer {i} restored parameters have success rate: {filepath['success_rate']}")
