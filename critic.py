import math
import torch
import torch.nn as nn


class Critic(nn.Module):

    def __init__(self, env, layer_number, FLAGS, device, gamma=0.98, tau=0.05):
        super(Critic, self).__init__()
        self.critic_name = 'critic_' + str(layer_number)
        self.gamma = gamma
        self.tau = tau
        self.device = device

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

        self.linear1 = nn.Linear(self.state_dim + action_dim + self.goal_dim, 64)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(64, 64)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(64, 1)

        # Set parameters to give critic optimistic initialization near q_init
        self.q_init = -0.067
        self.q_offset = -math.log(self.q_limit / self.q_init - 1)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                num_prev_neurons = int(m.weight.shape[1])
                fan_in_init = 1 / num_prev_neurons ** 0.5
                torch.nn.init.uniform_(m.weight, -fan_in_init, fan_in_init)
                torch.nn.init.uniform_(m.bias, -fan_in_init, fan_in_init)

        self.apply(init_weights)
        torch.nn.init.uniform_(self.output.weight, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.output, -3e-3, 3e-3)

    def forward(self, state, goal, action):
        h1 = self.relu1(self.linear1(torch.cat([state, goal, action], 1)))
        h2 = self.relu2(self.linear2(h1))
        h3 = self.output(h2)
        return torch.sigmoid(h3 + self.q_offset) * self.q_limit

    def update_target_weights(self, source):
        for target_param, param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def target_hard_init(self, source):
        for target_param, param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
