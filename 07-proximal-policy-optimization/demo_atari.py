from arguments import get_args
from ppo_agent import ppo_agent
import numpy as np
from baselines.common.cmd_util import make_atari_env
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from models import CNN_Net
import torch
import os

# get the tensors
def get_tensors(obs):
    return torch.tensor(np.transpose(obs, (0, 3, 1, 2)), dtype=torch.float32)

if __name__ == '__main__':
    args = get_args()
    # create the environment
    env = VecFrameStack(make_atari_env(args.env_name, 1, args.seed), 4)
    # start to create the model
    model_path = args.save_dir + args.env_name + '/model.pt'
    network = CNN_Net(env.action_space.n)
    network.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    # start to do the test
    obs = env.reset()
    for _ in range(10000):
        env.render()
        obs_tensor = get_tensors(obs)
        with torch.no_grad():
            _, pi = network(obs_tensor)
        actions = torch.argmax(pi, dim=1).item()
        obs, reward, done, _ = env.step([actions])
    env.close()
