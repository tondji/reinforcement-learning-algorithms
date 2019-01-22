import torch
import torch.nn as nn
import torch.nn.functional as F

# define the actor network
class Actor(nn.Module):
    def __init__(self, num_input, num_actions):
        super(Actor, self).__init__()
        self.affine_1 = nn.Linear(num_input, 64)
        self.affine_2 = nn.Linear(64, 64)
        self.actions = nn.Linear(64, num_actions)
        # start to do the initialization...
        nn.init.uniform_(self.actions.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.actions.bias, -3e-3, 3e-3)

    def forward(self, x):
        x = F.relu(self.affine_1(x))
        # the second layer
        x = F.relu(self.affine_2(x))
        # output the final actions
        policy = F.tanh(self.actions(x))

        return policy

# define the critic network...
class Critic(nn.Module):
    def __init__(self, num_input, num_actions):
        super(Critic, self).__init__()
        self.affine_1 = nn.Linear(num_input, 64)
        self.affine_2 = nn.Linear(64 + num_actions, 64)
        self.value = nn.Linear(64, 1)
        # start to do the initialization
        nn.init.uniform_(self.value.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.value.bias, -3e-3, 3e-3)

    def forward(self, x, actions):
        x = F.relu(self.affine_1(x))
        # the second layer..
        x = torch.cat((x, actions), 1)
        x = F.relu(self.affine_2(x))
        value = self.value(x)

        return value
