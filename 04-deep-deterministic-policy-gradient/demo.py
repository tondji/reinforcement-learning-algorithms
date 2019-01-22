from arguments import get_args
from ddpg_agent import ddpg_agent
import gym

if __name__ == '__main__':
    args = get_args()
    env = gym.make(args.env_name)
    ddpg_tester = ddpg_agent(args, env)
    ddpg_tester.test_network()
