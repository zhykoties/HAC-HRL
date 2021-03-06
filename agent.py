import numpy as np
from layer import Layer
import torch


# Below class instantiates an agent
class Agent:
    def __init__(self, FLAGS, env, agent_params):

        self.FLAGS = FLAGS

        # Set subgoal testing ratio each layer will use
        self.subgoal_test_perc = agent_params["subgoal_test_perc"]

        cuda_exist = torch.cuda.is_available()
        if cuda_exist:
            device = torch.device('cuda:0')
            print('Using CUDA...')
            torch.backends.cudnn.benchmark = True
        else:
            device = torch.device('cpu')
            print('Using cpu...')

        # Create agent with number of levels specified by user       
        self.layers = [Layer(i, FLAGS, env, agent_params, device) for i in range(FLAGS.layers)]
        self.num_layers = FLAGS.layers

        # Below attributes will be used help save network parameters
        self.saver = None
        self.model_dir = None
        self.model_loc = None

        # goal_array will store goal for each layer of agent.
        self.goal_array = [None for _ in range(FLAGS.layers)]

        self.current_state = None

        # Track number of low-level actions executed
        self.steps_taken = 0

        # Below hyperparameter specifies number of Q-value updates made after each episode
        self.num_updates = 40

        # Below parameters will be used to store performance results
        self.performance_log = []

        self.other_params = agent_params

    # Determine whether or not each layer's goal was achieved.  Also, if applicable, return the highest level whose
    # goal was achieved.
    def check_goals(self, env):

        # goal_status is vector showing status of whether a layer's goal has been achieved
        goal_status = [False for _ in range(self.FLAGS.layers)]

        max_lay_achieved = None

        # Project current state onto the subgoal and end goal spaces
        proj_subgoal = env.project_state_to_subgoal(env.sim, self.current_state)
        proj_end_goal = env.project_state_to_end_goal(env.sim, self.current_state)

        for i in range(self.FLAGS.layers):

            goal_achieved = True

            # If at highest layer, compare to end goal thresholds
            if i == self.FLAGS.layers - 1:

                # Check dimensions are appropriate         
                assert len(proj_end_goal) == len(self.goal_array[i]) == len(
                    env.end_goal_thresholds), "Projected end goal, actual end goal, and end goal thresholds should have same dimensions"

                # Check whether layer i's goal was achieved by checking whether projected state is within the goal
                # achievement threshold
                for j in range(len(proj_end_goal)):
                    if np.absolute(self.goal_array[i][j] - proj_end_goal[j]) > env.end_goal_thresholds[j]:
                        goal_achieved = False
                        break

            # If not highest layer, compare to subgoal thresholds
            else:

                # Check that dimensions are appropriate
                assert len(proj_subgoal) == len(self.goal_array[i]) == len(
                    env.subgoal_thresholds), "Projected subgoal, actual subgoal, and subgoal thresholds should have same dimensions"

                # Check whether layer i's goal was achieved by checking whether projected state is within the goal
                # achievement threshold
                for j in range(len(proj_subgoal)):
                    if np.absolute(self.goal_array[i][j] - proj_subgoal[j]) > env.subgoal_thresholds[j]:
                        goal_achieved = False
                        break

            # If projected state within threshold of goal, mark as achieved
            if goal_achieved:
                goal_status[i] = True
                max_lay_achieved = i
            else:
                goal_status[i] = False

        return goal_status, max_lay_achieved

    # Update actor and critic networks for each layer
    def learn(self):
        for i in range(len(self.layers)):
            self.layers[i].learn(self.num_updates)

    # Train agent for an episode
    def train(self, env, episode_num):

        # Select final goal from final goal space, defined in "design_agent_and_env.py" 
        self.goal_array[self.FLAGS.layers - 1] = env.get_next_goal(self.FLAGS.test)
        # print("Next End Goal: ", self.goal_array[self.FLAGS.layers - 1])

        # Select initial state from in initial state space, defined in environment.py
        self.current_state = env.reset_sim()
        # print("Initial State: ", self.current_state)

        # Reset step counter
        self.steps_taken = 0

        # Train for an episode
        goal_status, max_lay_achieved = self.layers[self.FLAGS.layers - 1].train(self, env, episode_num=episode_num)

        # Update actor/critic networks if not testing
        if not self.FLAGS.test:
            self.learn()

        # Return whether end goal was achieved
        return goal_status[self.FLAGS.layers - 1]
