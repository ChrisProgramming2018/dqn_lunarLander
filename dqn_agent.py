import os
import numpy as np
import random
import gym
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim
from replay_buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from model import QNetwork
import time
from gym import wrappers

from utils import time_format

class DQNAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, config):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = config["seed"]
        torch.manual_seed(self.seed)
        np.random.seed(seed=self.seed)
        random.seed(self.seed)
        self.env = gym.make(config["env_name"])
        self.env.seed(self.seed)
        self.env.action_space.seed(self.seed)
        self.lr = config['lr']
        self.batch_size = config['batch_size']
        self.device = config['device']
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.ddqn = config['DDQN']
        print("seed", self.seed)
        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, self.seed).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size, self.seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)
        self.memory = ReplayBuffer((state_size, ), (1, ), int(config["buffer_size"]), self.seed, config['device'])
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
        pathname = dt_string + "_use_double_" + str(self.ddqn) + "seed_" + str(config['seed'])
        tensorboard_name = str(config["locexp"]) + '/runs/' + pathname
<<<<<<< HEAD
        self.vid_path = str(config["locexp"]) + "/vid"
=======
        self.vid_path = str(config["locexp"])
>>>>>>> 62c9031c0fdc4e7db2962c76212c5f5e381ada59
        self.writer = SummaryWriter(tensorboard_name)
        self.steps = 0
        self.eps_decay = config["eps_decay"]
        self.eps_end = config["eps_end"]
        self.eps_start = config["eps_start"]
        self.episodes = config["episodes"]
        self.eval = config["eval"]
        self.locexp = str(config["locexp"])
<<<<<<< HEAD
        self.update_freq = config["update_freq"]
        self.agent = config["agent"]
        self.memory_size = config["memory_size"]

    def step(self):
        self.steps +=1 
        if self.steps % self.update_freq == 0:
            if len(self.memory) > self.batch_size:
                self.learn()
=======
    
    def step(self, memory):
        self.t_step +=1 
        if self.t_step % 4 == 0:
            if len(memory) > self.batch_size:
                self.learn(memory)
>>>>>>> 62c9031c0fdc4e7db2962c76212c5f5e381ada59

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
<<<<<<< HEAD
        self.steps += 1
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
=======
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

>>>>>>> 62c9031c0fdc4e7db2962c76212c5f5e381ada59
        # Get max predicted Q values (for next states) from target model
        if self.ddqn:
            local_actions = self.qnetwork_local(next_states).detach().max(1)[0]
            Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, local_actions)
        else:
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (self.gamma * Q_targets_next * dones)
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets.detach())
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.writer.add_scalar('q_loss', loss, self.steps)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def save_model(self, filename):
        os.makedirs(filename , exist_ok=True)
        torch.save(self.qnetwork_target.state_dict(), filename + "_critic")
        torch.save(self.optimizer.state_dict(), filename + "_critic_optimizer")

<<<<<<< HEAD
    def load_model(self, filename):
        self.qnetwork_local.load_state_dict(torch.load(filename + "_critic"))
        self.optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        print("... model loaded")

=======
>>>>>>> 62c9031c0fdc4e7db2962c76212c5f5e381ada59
    def train_agent(self):
        average_reward = 0
        scores_window = deque(maxlen=100)
        eps = self.eps_start
        t = 0
        t0 = time.time()
        for i_epiosde in range(1, self.episodes + 1):
            episode_reward = 0
            state = self.env.reset()
            while True:
                t += 1
                action = self.act(state, eps)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                if i_epiosde > 10:
<<<<<<< HEAD
                    self.step()
=======
                    self.learn()
>>>>>>> 62c9031c0fdc4e7db2962c76212c5f5e381ada59
                self.memory.add(state, action, reward, next_state, done, done)
                state = next_state
                if done:
                    break
            if i_epiosde % self.eval == 0:
                self.save_model(self.locexp +"/models/model-{}".format(self.steps))
                self.eval_policy()
            
            scores_window.append(episode_reward)
            eps = max(self.eps_end, self.eps_decay * eps) # decrease epsilon
            ave_reward = np.mean(scores_window)
            print("Epiosde {} Steps {} Reward {} Reward averge{:.2f}  eps {:.2f} Time {}".format(i_epiosde, t, episode_reward, np.mean(scores_window), eps, time_format(time.time() - t0)))
            self.writer.add_scalar('Aver_reward', ave_reward, self.steps)
            self.writer.add_scalar('steps_in_episode', t, self.steps)


    def act(self, state, eps):
        
        # Epsilon-greedy action selection
        if random.random() < eps:
            return random.choice(np.arange(self.action_size))
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        return np.argmax(action_values.cpu().data.numpy())


    def eval_policy(self, eval_episodes=4):
        env  = wrappers.Monitor(self.env, str(self.vid_path) + "/{}".format(self.steps), video_callable=lambda episode_id: True,force=True)
        average_reward = 0
        scores_window = deque(maxlen=100)
        for i_epiosde in range(eval_episodes):
<<<<<<< HEAD
            print("Eval Episode {} of {} ".format(i_epiosde, eval_episodes))
=======
            print("Eval Episode {} of {} ".format(i_epiosde, self.episodes))
>>>>>>> 62c9031c0fdc4e7db2962c76212c5f5e381ada59
            episode_reward = 0
            state = env.reset()
            while True:
                action = self.act(state, 0)
                state, reward, done, _ = env.step(action)
                episode_reward += reward
                if done:
                    scores_window.append(episode_reward)
                    break
        average_reward = np.mean(scores_window)
<<<<<<< HEAD
        print("Eval reward ", average_reward)
        self.writer.add_scalar('Eval_reward', average_reward, self.steps)

    def watch_trained_agent(self, eval_episodes):
        self.load_model(self.locexp + "/models/model-{}".format(self.agent))
        self.eval_policy(eval_episodes)
    
    def create_expert_memory(self):
        self.load_model(self.locexp + "/models/model-{}".format(self.agent))
        t = 0
        episode_reward = 0
        last_idx = 0
        while True:
            state = self.env.reset()
            while True:
                t += 1
                action = self.act(state, 0)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                self.memory.add(state, action, reward, next_state, done, done)
                if t >= self.memory_size:
                    self.memory.save_memory("expert_policy_size-{}".format(self.memory.idx))
                    return 
                state = next_state
                if done:
                    print("Episode Reward {} at memory size {}".format(episode_reward, self.memory.idx))
                    if episode_reward < 200:
                        self.memory.idx = last_idx
                        t = last_idx
                    else:
                        last_idx = self.memory.idx
                    episode_reward = 0
                    break
            

=======
        self.writer.add_scalar('Eval_reward', average_reward, self.steps)

>>>>>>> 62c9031c0fdc4e7db2962c76212c5f5e381ada59
