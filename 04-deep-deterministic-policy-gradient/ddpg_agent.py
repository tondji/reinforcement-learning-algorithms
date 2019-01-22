import numpy as np
import torch
from models import Actor, Critic
from ounoise import OUNoise
import random
from datetime import datetime
import os

# define the class...
class ddpg_agent:
    def __init__(self, args, env):
        self.args = args
        self.env = env
        # get the number of inputs...
        num_inputs = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.shape[0]
        self.action_scale = self.env.action_space.high[0]
        # build up the network
        self.actor_net = Actor(num_inputs, num_actions)
        self.critic_net = Critic(num_inputs, num_actions)
        # get the target network...
        self.actor_target_net = Actor(num_inputs, num_actions)
        self.critic_target_net = Critic(num_inputs, num_actions)
        if self.args.cuda:
            self.actor_net.cuda()
            self.critic_net.cuda()
            self.actor_target_net.cuda()
            self.critic_target_net.cuda()
        # copy the parameters..
        self.actor_target_net.load_state_dict(self.actor_net.state_dict())
        self.critic_target_net.load_state_dict(self.critic_net.state_dict())
        # setup the optimizer...
        self.optimizer_actor = torch.optim.Adam(self.actor_net.parameters(), lr=self.args.actor_lr)
        self.optimizer_critic = torch.optim.Adam(self.critic_net.parameters(), lr=self.args.critic_lr, weight_decay=self.args.critic_l2_reg)
        # setting up the noise
        self.ou_noise = OUNoise(num_actions)
        # check some dir
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        self.model_path = self.args.save_dir + self.args.env_name + '/'
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

    # start to train the network..
    def learn(self):
        # init the brain memory
        replay_buffer = []
        total_timesteps = 0
        running_reward = None
        for episode_idx in range(self.args.max_episode):
            state = self.env.reset()
            # get the scale of the ou noise...
            self.ou_noise.scale = (self.args.noise_scale - self.args.final_noise_scale) * max(0, self.args.exploration_length - episode_idx) / \
                                self.args.exploration_length + self.args.final_noise_scale
            self.ou_noise.reset()
            # start the training
            reward_total = 0
            while True:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                if self.args.cuda:
                    state_tensor = state_tensor.cuda()
                with torch.no_grad():
                    policy = self.actor_net(state_tensor)
                # start to select the actions...
                actions = self._select_actions(policy)
                # step
                state_, reward, done, _ = self.env.step(actions * self.action_scale)
                total_timesteps += 1
                reward_total += reward
                # start to store the samples...
                replay_buffer.append((state, reward, actions, done, state_))
                # check if the buffer size is outof range
                if len(replay_buffer) > self.args.replay_size:
                    replay_buffer.pop(0)
                if len(replay_buffer) > self.args.batch_size:
                    mini_batch = random.sample(replay_buffer, self.args.batch_size)
                    # start to update the network
                    _, _ = self._update_network(mini_batch)
                if done:
                    break
                state = state_
            running_reward = reward_total if running_reward is None else running_reward * 0.99 + reward_total * 0.01
            if episode_idx % self.args.display_interval == 0:
                torch.save(self.actor_net.state_dict(), self.model_path + 'model.pt')
                print('[{}] Episode: {}, Frames: {}, Rewards: {}'.format(datetime.now(), episode_idx, total_timesteps, running_reward))

        self.env.close()
    # select actions
    def _select_actions(self, policy):
        actions = policy.detach().cpu().numpy()[0]
        actions = actions + self.ou_noise.noise()
        actions = np.clip(actions, -1, 1)
        return actions
    
    # update the network
    def _update_network(self, mini_batch):
        state_batch = np.array([element[0] for element in mini_batch])
        state_batch = torch.tensor(state_batch, dtype=torch.float32)
        # reward batch
        reward_batch = np.array([element[1] for element in mini_batch])
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32).unsqueeze(1)
        # done batch
        done_batch = np.array([int(element[3]) for element in mini_batch])
        done_batch = 1 - done_batch
        done_batch = torch.tensor(done_batch, dtype=torch.float32).unsqueeze(1)
        # action batch
        actions_batch = np.array([element[2] for element in mini_batch])
        actions_batch = torch.tensor(actions_batch, dtype=torch.float32)
        # next stsate
        state_next_batch = np.array([element[4] for element in mini_batch])
        state_next_batch = torch.tensor(state_next_batch, dtype=torch.float32)
        # check if use the cuda
        if self.args.cuda:
            state_batch = state_batch.cuda()
            reward_batch = reward_batch.cuda()
            done_batch = done_batch.cuda()
            actions_batch = actions_batch.cuda()
            state_next_batch = state_next_batch.cuda()

        # update the critic network...
        with torch.no_grad():
            actions_out = self.actor_target_net(state_next_batch)
            expected_q_value = self.critic_target_net(state_next_batch, actions_out)
        # get the target value
        target_value = reward_batch + self.args.gamma * expected_q_value * done_batch
        target_value = target_value.detach()
        values = self.critic_net(state_batch, actions_batch)
        critic_loss = (target_value - values).pow(2).mean()
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()
        # start to update the actor network
        actor_loss = -self.critic_net(state_batch, self.actor_net(state_batch)).mean()
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()
        # then, start to softupdate the network...
        self._soft_update_target_network(self.critic_target_net, self.critic_net)
        self._soft_update_target_network(self.actor_target_net, self.actor_net)

        return actor_loss.item(), critic_loss.item()
    
    # soft update the network
    def _soft_update_target_network(self, target, source):
        # update the critic network firstly...
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
    
    # functions to test the network
    def test_network(self):
        model_path = self.args.save_dir + self.args.env_name + '/model.pt'
        self.actor_net.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        self.actor_net.eval()
        # start to test
        for _ in range(5):
            state = self.env.reset()
            reward_sum = 0
            while True:
                self.env.render()
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    actions = self.actor_net(state)
                actions = actions.detach().numpy()[0]
                state_, reward, done, _ = self.env.step(self.action_scale * actions)
                reward_sum += reward
                if done:
                    break
                state = state_
            print('The reward of this episode is {}.'.format(reward_sum))
        self.env.close()
