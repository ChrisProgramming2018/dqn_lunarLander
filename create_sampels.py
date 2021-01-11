import gym
import random
import torch
import numpy as np
import argparse

from dqn_agent import DQNAgent
from replay_buffer2 import ReplayBuffer
from iql_agent import mkdir


def man(args):
    env = gym.make(args.env_name)
    env.seed(args.seed)
    state_size = env.observation_space.shape)
    action_size = env.action_space.n)
    print('State shape: ', state_size)
    print('Number of actions: ', action_size)
    agent = DQNAgent(state_size=state_size, action_size=action_size, seed=args.seed)
    agent.qnetwork_local.load_state_dict(torch.load(args.modelpath))
    memory = ReplayBuffer((state_size,), (1,), args.max_steps, args.device)
    total_maxsteps = args.max_steps
    eps = 0
    rewards = []
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        while True:
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            score += reward
            memory.add(state, action, reward, next_state, done, done)
            state = next_state
            if done:
                rewards.append(score)
                print("Episode {}  Reward {}".format(i_episode, score))
                break
    mean_rewards = np.mean(rewards)
    print("Episode {} average Reward {}".format(i_episode, score))

    mkdir("","expert_policy")
    print("save memory ...")
    memory.save_memory("expert_policy")
    print("... memory saved")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default="LunarLander-v2", type=str)
    parser.add_argument('--modelpath', default="", type=str)
    parser.add_argument('--device', default="cuda", type=str)
    arg = parser.parse_args()
    main(arg)
