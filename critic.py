import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):

    def __init__(self, env, layer_number, FLAGS, gamma=0.98, tau=0.05):
        super(Critic, self).__init__()
        self.critic_name = 'critic_' + str(layer_number)
        self.gamma = gamma
        self.tau = tau

        self.q_limit = -FLAGS.time_scale

        # Dimensions of goal placeholder will differ depending on layer level
        if layer_number == FLAGS.layers - 1:
            self.goal_dim = env.end_goal_dim
        else:
            self.goal_dim = env.subgoal_dim

        self.loss_val = 0
        self.state_dim = env.state_dim

        # Dimensions of action placeholder will differ depending on layer level
        if layer_number == 0:
            action_dim = env.action_dim
        else:
            action_dim = env.subgoal_dim

        self.critic = nn.Sequential(
                        nn.Linear(env.state_dim + action_dim + self.goal_dim, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, 1)
                        )

        # Set parameters to give critic optimistic initialization near q_init
        self.q_init = -0.067
        self.q_offset = -math.log(self.q_limit / self.q_init - 1)

    def forward(self, state, goal, action):
        return F.sigmoid(self.critic(torch.cat([state, goal, action], 1)) + self.q_offset) * self.q_limit

    def update_target_weights(self, source):
        for target_param, param in zip(self.critic.parameters(), source.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def target_hard_init(self, source):
        for target_param, param in zip(self.critic.parameters(), source.critic.parameters()):
            target_param.data.copy_(param.data)
