import argparse

"""
Below are training options user can specify in command line.

Options Include:

1. Retrain boolean
- If included, actor and critic neural network parameters are reset

2. Testing boolean
- If included, agent only uses greedy policy without noise.  No changes are made to policy and neural networks. 
- If not included, periods of training are by default interleaved with periods of testing to evaluate progress.

3. Show boolean
- If included, training will be visualized

4. Train Only boolean
- If included, agent will be solely in training mode and will not interleave periods of training and testing

5. Verbosity boolean
- If included, summary of each transition will be printed
"""


def parse_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('--env',
                        default='ur5',
                        help='Name of the environment')

    parser.add_argument('--model',
                        default='LAY_3',
                        help='Model settings')

    parser.add_argument('--exp_name',
                        default=None,
                        help='Name of experiment (optional)')

    parser.add_argument('--retrain',
                        action='store_true',
                        help='Include to reset policy')

    parser.add_argument('--test',
                        action='store_true',
                        help='Include to fix current policy')

    parser.add_argument('--show',
                        action='store_true',
                        help='Include to visualize training')

    parser.add_argument('--train_only',
                        action='store_true',
                        help='Include to use training mode only')

    parser.add_argument('--verbose',
                        action='store_true',
                        help='Print summary of each transition')

    parser.add_argument('--lvl_parallel',
                        action='store_true',
                        help='Experiment: test each level using different checkpoints')

    parser.add_argument('--max_subgoal_explore',
                        default=0,
                        type=int,
                        help='Name of experiment (optional)')

    parser.add_argument(
        '--restore-file',
        default=None,
        help='Optional, name of file in model_dir containing weights to reload before training')  # 'last' or 'epoch_#'

    FLAGS, unparsed = parser.parse_known_args()

    return FLAGS
