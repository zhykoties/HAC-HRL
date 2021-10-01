import torch
import torch.nn as nn


class Actor(nn.Module):

    def __init__(self, env, layer_number, FLAGS, device, tau=0.05):
        super(Actor, self).__init__()
        self.device = device

        # Determine range of actor network outputs.  This will be used to configure outer layer of neural network
        if layer_number == 0:
            self.action_space_bounds = torch.tensor(env.action_bounds, dtype=torch.float32, device=device)
            self.action_offset = torch.tensor(env.action_offset, dtype=torch.float32, device=device)
        else:
            # Determine symmetric range of subgoal space and offset
            self.action_space_bounds = torch.tensor(env.subgoal_bounds_symmetric, dtype=torch.float32, device=device)
            self.action_offset = torch.tensor(env.subgoal_bounds_offset, dtype=torch.float32, device=device)

        # Dimensions of action will depend on layer level     
        if layer_number == 0:
            self.action_space_size = env.action_dim
        else:
            self.action_space_size = env.subgoal_dim

        self.actor_name = 'actor_' + str(layer_number)

        if layer_number == FLAGS.layers - 1:
            self.goal_dim = env.end_goal_dim
        else:
            self.goal_dim = env.subgoal_dim

        self.state_dim = env.state_dim

        # self.exploration_policies = exploration_policies
        self.tau = tau
        self.actor = nn.Sequential(
            nn.Linear(self.state_dim + self.goal_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_space_size),
            nn.Tanh()
        )

    def forward(self, state, goal):
        return self.actor(torch.cat([state, goal], 1)) * self.action_space_bounds + self.action_offset

    def update_target_weights(self, source):
        for target_param, param in zip(self.actor.parameters(), source.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def target_hard_init(self, source):
        for target_param, param in zip(self.actor.parameters(), source.actor.parameters()):
            target_param.data.copy_(param.data)
