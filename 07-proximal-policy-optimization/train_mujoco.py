from arguments import get_args
from ppo_agent import ppo_agent
from models import MLP_Net
from baselines.common.cmd_util import make_mujoco_env
from baselines import logger
import os

if __name__ == '__main__':
    args = get_args()
    if not os.path.exists('logs/'):
        os.mkdir('logs/')
    log_path = 'logs/' + args.env_name + '/'
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    # write log information
    logger.configure(log_path)
    env = make_mujoco_env(args.env_name, args.seed)
    #env = gym.make(args.env_name)
    network = MLP_Net(env.observation_space.shape[0], env.action_space.shape[0], args.dist)
    ppo_trainer = ppo_agent(env, args, network, 'mujoco')
    ppo_trainer.learn()
