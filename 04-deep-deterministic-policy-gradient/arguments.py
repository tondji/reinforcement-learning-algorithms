import argparse

# get the arguments...
def get_args():
    parse = argparse.ArgumentParser(description='ddpg')
    parse.add_argument('--env-name', type=str, default='Pendulum-v0', help='the training environment')
    parse.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parse.add_argument('--tau', type=float, default=0.001, help='discount factor')
    parse.add_argument('--noise-scale', type=float, default=0.3, help='noise scale')
    parse.add_argument('--final-noise-scale', type=float, default=0.3, help='final noise scale')
    parse.add_argument('--exploration-length', type=int, default=100, help='the episode that end the exploration')
    parse.add_argument('--seed', type=int, default=123, help='the random seed')
    parse.add_argument('--batch-size', type=int, default=128, help='the batch size that sample')
    parse.add_argument('--max-steps', type=int, default=1000, help='the max time steps per episode')
    parse.add_argument('--max-episode', type=int, default=1000, help='the max length of episode to train the agent')
    parse.add_argument('--updates-per-step', type=int, default=1, help='num of update per steps')
    parse.add_argument('--replay-size', type=int, default=1000000, help='the size of replay buffer')
    parse.add_argument('--cuda', action='store_true', help='if use the GPU to do the training')
    parse.add_argument('--actor-lr', type=float, default=1e-4, help='the learning rate of actor network')
    parse.add_argument('--critic-lr', type=float, default=1e-3, help='the learning rate of critic network')
    parse.add_argument('--critic-l2-reg', type=float, default=1e-2, help='the reg coefficient')
    parse.add_argument('--display-interval', type=int, default=1, help='display interval')
    parse.add_argument('--save-dir', type=str, default='saved_models/', help='the folder that save the models')

    # get args...
    args = parse.parse_args()

    return args
