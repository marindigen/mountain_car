""""0. initialize libraries and setup tools"""

import os
import numpy as np
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
import json

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" 1.1. Helper functions """
    
def plot_durations(episode_durations, run_id):
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Duration per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.scatter(list(range(len(episode_durations))), durations_t.numpy(), color='grey')
    
    moving_avg = []
    for i in range(len(episode_durations)):
        if i < 100:
            moving_avg.append(durations_t[:i+1].mean().item())  
        else:
            moving_avg.append(durations_t[i-99:i+1].mean().item()) 
    
    plt.plot(range(len(episode_durations)), moving_avg, color='orange', label='100-Moving Average')
    plt.legend()
    plt.savefig(f"DQN_Duration_Hyperparameters_{run_id}.png", dpi=300)
    plt.close()
    
def plot_curve(data, title, xlabel, ylabel, run_id):
    plt.plot(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(f"DQN_{title}_H{run_id}.png", dpi=300)
    plt.close()

def plot_cumul_reward(data, xlabel, ylabel, run_id):
    plt.figure(1)
    data_t = torch.tensor(data, dtype=torch.float)
    plt.title('Cumulative Reward Per Episode')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.scatter(list(range(len(data))), data_t.numpy(), color = 'grey', label='Cumulative Reward Per Episode') 
    if len(data_t) > 100:
        means = data_t[100:].unfold(0, 100, 1).mean(1).view(-1)
        plt.plot(range(100, len(data)-99), means.numpy(), color = 'orange', label='Moving Average (100 episodes)')
    plt.legend()
    plt.savefig(f"DQN_Cumul_Reward_{run_id}.png", dpi=300)
    plt.close()
    
def cumulative_sum(input_list):
    result = []
    running_total = 0
    for element in input_list:
        running_total += element
        result.append(running_total)
    return result

def plot_comp_cumul_reward(data, data_env, data_aux, run_id=1):
    data_t = torch.tensor(data, dtype=torch.float)
    plt.title('Composition of Averaged Cumulative Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    c_e_r_p_e_t = torch.tensor(data_env, dtype=torch.float)
    c_a_r_p_e_t = torch.tensor(data_aux, dtype=torch.float)
    if len(c_e_r_p_e_t) > 100:  # Check if data has more than 100 episodes
            # Calculate and plot the average line starting from episode 100
            means = data_t[100:].unfold(0, 100, 1).mean(1).view(-1)
            plt.plot(range(100, len(data)-99), means.numpy(), color='cyan', label='Moving Average Reward Per Episode')
            plt.scatter(torch.arange(len(data_t)), data_t, s = 2, color='blue', label='Cumulative Reward Per Episode')
            # Calculate and plot moving average starting from episode 100
            means = c_e_r_p_e_t[100:].unfold(0, 100, 1).mean(1).view(-1)
            plt.plot(range(100, len(data_env)-99), means.numpy(), color='yellow', label='Moving Average Environment Reward Per Episode')
            plt.scatter(torch.arange(len(data_t)), c_e_r_p_e_t, s=2, color='goldenrod', label='Cumulative Environment Reward Per Episode')
            # Plot the average line starting from episode 100
            means = c_a_r_p_e_t[100:].unfold(0, 100, 1).mean(1).view(-1)
            plt.plot(range(100, len(data_aux)-99), means.numpy(), color='springgreen', label='Moving Average Intrinsic Reward Per Episode')
            plt.scatter(torch.arange(len(data_t)), c_a_r_p_e_t, s=2, color='forestgreen',label='Cumulative Intrinsic Reward Per Episode')
    plt.legend()
    plt.savefig(f"DQN_Comp_Cumul_Reward_{run_id}.png", dpi=300)
    plt.close()

def plot_evaluation(episode_durations, run_id):
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Duration per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.scatter(list(range(len(episode_durations))), durations_t.numpy(), color='grey')
    
    moving_avg = []
    for i in range(len(episode_durations)):
        if i < 100:
            moving_avg.append(durations_t[:i+1].mean().item())  
        else:
            moving_avg.append(durations_t[i-99:i+1].mean().item()) 
    
    plt.plot(range(len(episode_durations)), moving_avg, color='orange', label='100-Moving Average')
    plt.legend()
    plt.savefig(f"DQN_Evaluation_Hyperparameters_{run_id}.png", dpi=300)
    plt.close()

'''Classes for DQN and RND + buffer'''
# We have also set up a RunningStats class for more clear processing of the means and standard deviations for both rewards and states.

""""1.2. Define ReplayBuffer and deep q network"""

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'reward_int'))

class ReplayBuffer(object):

    def __init__(self, capacity):
        self.buffer = deque([], maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args)) #Save a transition

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions, n_nodes_per_layer=64, n_layers=2):
        super(DQN, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(n_observations, n_nodes_per_layer)])
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(n_nodes_per_layer, n_nodes_per_layer))
        self.output_layer = nn.Linear(n_nodes_per_layer, n_actions)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.output_layer(x)
    
class RNDNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, n_hid):
        super(RNDNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, n_hid),
            nn.ReLU(),
            nn.Linear(n_hid, n_hid),
            nn.ReLU(),
            nn.Linear(n_hid, output_dim),
        )

    def forward(self, x):
        return self.network(x)
            
class RND:
    def __init__(self,in_dim,out_dim,n_hid):
        self.target = RNDNetwork(in_dim,out_dim,n_hid)
        self.predictor = RNDNetwork(in_dim,out_dim,n_hid)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(),lr=0.0001)
        
    def get_reward(self, x):
        y_true = self.target(x).detach()  # Detaching to stop gradients
        y_pred = self.predictor(x)
        #reward = torch.pow(y_pred - y_true, 2).sum()
        reward = nn.MSELoss()(y_pred, y_true).to(device)
        return reward, y_pred

    def update(self, x):
        self.optimizer.zero_grad()
        r,__= self.get_reward(x)
        loss = r
        loss.backward()  # Compute gradients
        self.optimizer.step()
    

class RunningStats:
    def __init__(self, dimension=None, device='cpu'):
        self.n = 0
        self.mean = None
        self.M2 = 0
        self.dimension = dimension
        self.std = 1
        self.device = device

    def update(self, x):
        if torch.is_tensor(x) == False:
            x = torch.tensor(x, device=self.device)  # Convert to tensor if not already

        if self.mean is None:
            self.mean = torch.zeros_like(x, device=self.device)
            self.M2 = torch.zeros_like(x, device=self.device)

        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def variance(self):
        if self.n < 2:
            return torch.zeros_like(self.mean)
        return self.M2 / (self.n - 1)

    def std_dev(self):
        self.std = torch.sqrt(self.variance())
        return self.std

    @property
    def get_mean(self):
        return self.mean

    @property
    def get_std(self):
        return self.std
    
''' Agent class '''
# Includes the evaluation method for the comaprison part 
"""" 2. Agent class """

class DQNAgent:
    def __init__(self, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR, n_actions, n_observations):
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.TAU = TAU
        self.LR = LR
        self.n_actions = n_actions
        self.n_observations = n_observations
        self.memory = ReplayBuffer(10000)
        self.policy_net = DQN(n_observations, n_actions).to(device)
        self.target_net = DQN(n_observations, n_actions).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.steps_done = 0
        self.episode_durations = []
        self.loss_per_episode = []
        self.cumulative_reward_per_episode = []
        self.cumulative_environment_reward_per_episode = []
        self.cumulative_intrinsic_reward_per_episode = []
        self.agent_performance = []
        self.testing_durations = []

        self.reward_factor = 2.5
        #self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        #self.predictor_net = RNDNetwork(n_observations, 1).to(device)
        #self.target_rnd_net = RNDNetwork(n_observations, 1).to(device)
        self.rnd = RND(n_observations,1,128)

        # Running statistics for normalization
        self.state_stats = RunningStats()
        self.reward_stats = RunningStats()


    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        reward_int_batch = torch.cat([r[0].unsqueeze(0) for r in batch.reward_int if r is not None])
        #reward_int_batch_norm = self.normalise_reward(reward_int_batch).detach()
        
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        #for ri in reward_int_batch:
        #    self.reward_stats.update(ri.detach())
        #print("Total reward:", torch.sum(reward_batch + ((reward_int_batch.detach()-self.reward_stats.mean)/self.reward_stats.std).clamp(-5,5) * self.reward_factor))
        #print("Just intrinsic reward:", ((reward_int_batch.detach()-self.reward_stats.mean)/self.reward_stats.std).clamp(-5,5))
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch + ((reward_int_batch.detach()-self.reward_stats.mean)/self.reward_stats.std).clamp(-5,5) * self.reward_factor
        expected_state_action_values = expected_state_action_values.float()

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        #for state in state_batch:
        #    self.state_stats.update(state)
        state_batch_norm = (state_batch-self.state_stats.mean)/self.state_stats.std
        self.rnd.update(state_batch_norm)

        return loss.item()

    def update(self, num_episodes, M):
        preliminary_r_i = []
        for m in range(M):
                state, info = env.reset()
                state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = env.step(action.item())
                observation = torch.tensor(observation, dtype=torch.float32, device=device)
                self.state_stats.update(observation)
                state_norm = (observation-self.state_stats.mean)/self.state_stats.std
                reward_int = self.rnd.get_reward(state_norm)
                preliminary_r_i.append(reward_int[0])
        self.reward_stats.mean = torch.mean(torch.tensor(preliminary_r_i)).to(device)
        self.reward_stats.std = torch.std(torch.tensor(preliminary_r_i)).to(device)
        for episode in range(num_episodes):
            state, info = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            cumulative_reward_t = 0
            environment_reward_t = 0
            intrinsic_reward_t = 0
            running_loss = []
            for t in count():
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = env.step(action.item())
                environment_reward_t += reward
                reward = torch.tensor([reward], device=device)
                observation = torch.tensor(observation, dtype=torch.float32, device=device)

                state_norm = (observation-self.state_stats.mean)/self.state_stats.std
                reward_int = self.rnd.get_reward(state_norm)
                intrinsic_reward_t += ((reward_int[0] - self.reward_stats.mean)/self.reward_stats.std + 1e-8).clamp(-5,5)
                cumulative_reward_t += environment_reward_t + intrinsic_reward_t.detach()
                self.state_stats.update(observation)
                self.reward_stats.update(reward_int[0].detach())
                done = terminated or truncated
                
                if not done:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                else:
                    next_state = None
                if truncated:
                    self.agent_performance.append(0)
                
                #print("Extrinsic reward:", reward)
                #print("Intrinsic reward:", reward_int)
                self.memory.push(state, action, next_state, reward, reward_int)
                #self.intrinsic_rewards.append(reward_int)

                state = next_state
                
                loss_value = self.optimize_model()
                
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    self.episode_durations.append(t + 1)
                    if terminated:
                        self.agent_performance.append(1)
                    else:
                        self.agent_performance.append(0)
                    print(f"Episode {episode}: episode duration {self.episode_durations[-1]}")
                    break
                    
                running_loss.append(loss_value)
                
            filtered_running_loss = [loss for loss in running_loss if loss is not None]
            if filtered_running_loss:
                self.loss_per_episode.append(np.mean(filtered_running_loss))

            self.cumulative_reward_per_episode.append(cumulative_reward_t)
            self.cumulative_environment_reward_per_episode.append(environment_reward_t)
            self.cumulative_intrinsic_reward_per_episode.append(intrinsic_reward_t.detach())

        return torch.save(self.policy_net.state_dict(), 'rnd_dqn_policynet.pth')
    
    def evaluate(self, modelfile, seedfile, num_episodes):
        self.policy_net.load_state_dict(torch.load(modelfile))
        with open(seedfile, 'r') as f:
            seeds = json.load(f)
        for i_episode in tqdm(range(num_episodes), desc="Episodes"):
            env.reset()
            state = seeds[i_episode]
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            for t in count():
                with torch.no_grad():
                    action = self.policy_net(state).max(1)[1].view(1, 1)
                observation, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated
                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                state = next_state
                if done:
                    self.testing_durations.append(t + 1)
                    break

def train_agent_with_hyperparameters(BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR, run_id, num_episodes=3000):
    n_actions = env.action_space.n
    n_observations = env.observation_space.shape[0]
    agent = DQNAgent(BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR, n_actions, n_observations)
    agent.update(num_episodes, M=10)
    hyperparameters = f"BATCH_SIZE={BATCH_SIZE}_GAMMA={GAMMA}_EPS_START={EPS_START}_EPS_END={EPS_END}_EPS_DECAY={EPS_DECAY}_TAU={TAU}_LR={LR}"
    
    # Save the list to a file
    with open('data_rnd/rnd_episode_durations.json', 'w') as f:
        json.dump(agent.episode_durations, f)
    with open('data_rnd/rnd_cumulative_environment_reward_per_episode.json', 'w') as f:
        json.dump(agent.cumulative_environment_reward_per_episode, f)

    plot_durations(agent.episode_durations, run_id)
    plot_curve(agent.loss_per_episode, 'Loss Curve', 'Training Step', 'Loss', run_id)
    plot_cumul_reward(torch.tensor(agent.cumulative_reward_per_episode)/torch.tensor(agent.episode_durations), 'Episode', 'Cumulative Reward', run_id=20)
    plot_comp_cumul_reward(torch.tensor(agent.cumulative_reward_per_episode)/torch.tensor(agent.episode_durations), agent.cumulative_environment_reward_per_episode, agent.cumulative_intrinsic_reward_per_episode, run_id=20)
    cumulative_successes = cumulative_sum(agent.agent_performance)
    plot_curve(cumulative_successes, 'Cumulative Number of Successes', 'Episodes', 'Successes', run_id)
    return agent.episode_durations, agent.cumulative_reward_per_episode, agent.cumulative_environment_reward_per_episode, agent.cumulative_intrinsic_reward_per_episode, agent.agent_performance

def evaluate_agent(modelfile, seedfile, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR, run_id, num_episodes=1000):
    n_actions = env.action_space.n
    n_observations = env.observation_space.shape[0]
    agent = DQNAgent(BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR, n_actions, n_observations)
    agent.evaluate(modelfile, seedfile, num_episodes)
    hyperparameters = f"BATCH_SIZE={BATCH_SIZE}_GAMMA={GAMMA}_EPS_START={EPS_START}_EPS_END={EPS_END}_EPS_DECAY={EPS_DECAY}_TAU={TAU}_LR={LR}"
    
    with open('data_rnd/rnd_testing_durations.json', 'w') as f:
        json.dump(agent.testing_durations, f)

    plot_evaluation(agent.testing_durations, run_id)


if __name__ == '__main__':
    # Set up the environment
    env = gym.make('MountainCar-v0')

    # set up matplotlib
    is_ipython = 'inline' in matplotlib.get_backend()  # checks if one is running IPhyton environment like jupyter notebook
    if is_ipython:
        from IPython import display
    plt.ion()  # interactive mode on, allows automatic plots when data is updated (whithout calling plt.show every time)

    # set the run id
    run_id = 1

    episode_durations,cum_r, cum_env_r, cum_int_r, agent_performance = train_agent_with_hyperparameters(64, 0.99, 0.9, 0.1, 100000, 0.0005, run_id, 1e-4)
    evaluate_agent('model_rnd/rnd_dqn_policynet.pth', 'agent_comparison/seeds.json', 64, 0.99, 0.9, 0.1, 100000, 0.0001, 1e-4, run_id, 1000)