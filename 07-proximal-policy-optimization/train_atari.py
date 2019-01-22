from arguments import get_args
from ppo_agent import ppo_agent
from baselines.common.cmd_util import make_atari_env
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from models import CNN_Net
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
    logger.configure(dir=log_path)
    envs = VecFrameStack(make_atari_env(args.env_name, args.num_workers, args.seed), 4)
    network = CNN_Net(envs.action_space.n)
    ppo_trainer = ppo_agent(envs, args, network, 'atari')
    ppo_trainer.learn()
