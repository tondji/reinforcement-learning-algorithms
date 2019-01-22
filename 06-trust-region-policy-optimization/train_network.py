import gym
from arguments import get_args
import os
from baselines import logger
from baselines.common.cmd_util import make_mujoco_env
from trpo_agent import trpo_agent

if __name__ == '__main__':
    args = get_args()
    if not os.path.exists('logs/'):
        os.mkdir('logs/')
    log_path = 'logs/' + args.env_name + '/'
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    logger.configure(log_path)
    # make environemnts
    env = make_mujoco_env(args.env_name, args.seed)
    trpo_trainer = trpo_agent(env, args)
    trpo_trainer.learn()
