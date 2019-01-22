from baselines.common.cmd_util import make_atari_env
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from arguments import get_args
from a2c_agent import a2c_agent
from baselines import logger

if __name__ == '__main__':
    args = get_args()
    logger.configure(dir=args.log_dir)
    # create environments
    envs = VecFrameStack(make_atari_env(args.env_name, args.num_processes, args.seed), 4)
    trainer = a2c_agent(envs, args)
    trainer.learn()
    envs.close()
