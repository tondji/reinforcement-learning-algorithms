import torch
from torch import nn
from torch.nn import functional as F

class Network(nn.Module):
    def __init__(self, num_states, num_actions):
        super(Network, self).__init__()
        # define the critic
        self.critic = Critic(num_states)
        self.actor = Actor(num_states, num_actions)

    def forward(self, x):
        state_value = self.critic(x)
        pi = self.actor(x)
        return state_value, pi

class Critic(nn.Module):
    def __init__(self, num_states):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_states, 64)
        self.fc2 = nn.Linear(64, 64)
        self.value = nn.Linear(64, 1)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        value = self.value(x)
        return value

class Actor(nn.Module):
    def __init__(self, num_states, num_actions):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_states, 64)
        self.fc2 = nn.Linear(64, 64)
        self.action_mean = nn.Linear(64, num_actions)
        self.sigma_log = nn.Parameter(torch.zeros(1, num_actions))

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        mean = self.action_mean(x)
        sigma_log = self.sigma_log.expand_as(mean)
        sigma = torch.exp(sigma_log)
        pi = (mean, sigma)
        
        return pi
