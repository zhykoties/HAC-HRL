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
        self.linear1 = nn.Linear(self.state_dim + self.goal_dim, 64)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(64, 64)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(64, self.action_space_size)
        self.tanh = nn.Tanh()

        def init_weights(m):
            if isinstance(m, nn.Linear):
                num_prev_neurons = int(m.weight.shape[1])
                fan_in_init = 1 / num_prev_neurons ** 0.5
                torch.nn.init.uniform_(m.weight, -fan_in_init, fan_in_init)
                torch.nn.init.uniform_(m.bias, -fan_in_init, fan_in_init)

        self.apply(init_weights)
        torch.nn.init.uniform_(self.output.weight, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.output.bias, -3e-3, 3e-3)

    def forward(self, state, goal):
        h1 = self.relu1(self.linear1(torch.cat([state, goal], 1)))
        h2 = self.relu2(self.linear2(h1))
        h3 = self.tanh(self.output(h2))
        return h3 * self.action_space_bounds + self.action_offset

    def update_target_weights(self, source):
        for target_param, param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def target_hard_init(self, source):
        for target_param, param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
