import random
import gym
import torch
import numpy as np
import argparse
import json
from collections import namedtuple, deque
from dqn_agent import DQNAgent




def main(args):
    print("start in {} mode ".format(args.mode))
    with open (args.param, "r") as f:
        config = json.load(f)
    config["seed"] = args.seed
    config["agent"] = args.agent
    config["memory_size"] = args.memory_size
    config["track"] = args.track
    config["wandb_name"] = args.wandb_name
    if args.game is not None:
        print("change game ", args.game)
        config["env_name"] = args.game
    env = gym.make(config["env_name"])
    #env = gym.wrappers.Monitor(env, "./vid", video_callable=lambda episode_id: True,force=True)
    env.seed(config['seed'])
    state_size = env.observation_space.shape[0]
    print('State shape: ', state_size)
    action_size = env.action_space.n
    print('Number of actions: ', action_size)
    agent = DQNAgent(state_size=state_size, action_size=action_size, config=config)
    if args.mode == "train":
        agent.train_agent()
    if args.mode == "eval":
        agent.watch_trained_agent(20)
    if args.mode == "m":
        agent.create_expert_memory()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--param', default="param.json", type=str)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--fc1_units', default=256, type=int)
    parser.add_argument('--fc2_units', default=256, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--mode', default="train", type=str)
    parser.add_argument('--buffer_size', default=3e5, type=int)
    parser.add_argument('--max_episode_steps', default=1000, type=int) 
    parser.add_argument('--agent', default=1, type=int) 
    parser.add_argument('--memory_size', default=5000, type=int) 
    parser.add_argument('--locexp', default="", type=str) 
    parser.add_argument('--game', default=None, type=str) 
    parser.add_argument('--track', default=False, type=bool) 
    parser.add_argument('--wandb_name', default="test", type=str) 
    arg = parser.parse_args()
    main(arg)

