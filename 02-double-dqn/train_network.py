from arguments import get_args
from baselines.common.atari_wrappers import make_atari
from baselines import bench
from baselines import logger
from baselines.common.atari_wrappers import wrap_deepmind
from ddqn_agent import ddqn_agent
import os 

if __name__ == '__main__':
    args = get_args()
    if not os.path.exists('logs/'):
        os.mkdir('logs/')
    log_path = 'logs/' + args.env_name + '/'
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    logger.configure(log_path)
    # start to create the environment
    env = make_atari(args.env_name)
    env = bench.Monitor(env, logger.get_dir())
    env = wrap_deepmind(env, frame_stack=True)
    trainer = ddqn_agent(env, args)
    trainer.learn()
    env.close()
