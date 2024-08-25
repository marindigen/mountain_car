""""0. initialize libraries and setup tools"""

import numpy as np
import gymnasium as gym
import math
import random
import json
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
from torchsummary import summary



# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    
    
    
""" 1.1. Helper functions """
    
def plot_durations(episode_durations, hyperparameters):
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
    plt.savefig(f"DQN_Duration_Hyperparameters_{hyperparameters}.png", dpi=300)
    plt.close()
    
def plot_evaluation(episode_durations, hyperparameters):
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
    plt.savefig(f"DQN_Evaluation_Hyperparameters_{hyperparameters}.png", dpi=300)
    plt.close()
    
def plot_curve(data, title, xlabel, ylabel, hyperparameters):
    plt.plot(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(f"DQN_{title}_Hyperparameters_{hyperparameters}.png", dpi=300)
    plt.close()

def plot_cumul_reward(data, xlabel, ylabel, hyperparameters):
    plt.figure(1)
    data_t = torch.tensor(data, dtype=torch.float)
    plt.title('Cumulative Reward Per Episode')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.scatter(list(range(len(data))), data_t.numpy(), color = 'grey') 
    if len(data_t) > 100:
        means = data_t[100:].unfold(0, 100, 1).mean(1).view(-1)
        plt.plot(range(100, len(data)-99), means.numpy(), color = 'orange', label='Moving Average (100 episodes)')
    plt.legend()
    plt.savefig(f"DQN_Cumul_Reward_Hyperparameters_{hyperparameters}.png", dpi=300)
    plt.close()
    
def cumulative_sum(input_list):
    result = []
    running_total = 0
    for element in input_list:
        running_total += element
        result.append(running_total)
    return result

def plot_comp_cumul_reward_episode(data, data_env, data_aux, hyperparameters):
    data_t = torch.tensor(data, dtype=torch.float)
    plt.title('Composition of Averaged Cumulative Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    c_e_r_p_e_t = torch.tensor(data_env, dtype=torch.float)
    c_a_r_p_e_t = torch.tensor(data_aux, dtype=torch.float)
    if len(c_e_r_p_e_t) > 100:  # Check if data has more than 100 episodes
            means = data_t[100:].unfold(0, 100, 1).mean(1).view(-1)
            # Plot the average line starting from episode 100
            plt.plot(range(100, len(data)-99), means.numpy(), label='Cumulative Reward Per Episode')
            # Calculate moving average starting from episode 100
            means = c_e_r_p_e_t[100:].unfold(0, 100, 1).mean(1).view(-1)
            # Plot the average line starting from episode 100
            plt.plot(range(100, len(data_env)-99), means.numpy(), label='Cumulative Environment Reward Per Episode')
            means = c_a_r_p_e_t[100:].unfold(0, 100, 1).mean(1).view(-1)
            # Plot the average line starting from episode 100
            plt.plot(range(100, len(data_aux)-99), means.numpy(), label='Cumulative Auxiliary Reward Per Episode')
    plt.legend()
    plt.savefig(f"DQN_Comp_Cumul_Reward_per_episode_Hyperparameters_{hyperparameters}.png", dpi=300)
    plt.close()

def plot_comp_cumul_reward(data, data_env, data_aux, hyperparameters):
    cumulative_rewards = cumulative_sum(data)
    cumulative_env_rewards = cumulative_sum(data_env)
    cumulative_aux_rewards = cumulative_sum(data_aux)

    data_t = torch.tensor(cumulative_rewards, dtype=torch.float)
    plt.title('Composition of Averaged Cumulative Reward')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    c_e_r_t = torch.tensor(cumulative_env_rewards, dtype=torch.float)
    c_a_r_t = torch.tensor(cumulative_aux_rewards, dtype=torch.float)
    if len(c_e_r_t) > 100:  # Check if data has more than 100 episodes
            means = data_t[100:].unfold(0, 100, 1).mean(1).view(-1)
            # Plot the average line starting from episode 100
            plt.plot(range(100, len(data)-99), means.numpy(), label='Cumulative Reward')
            # Calculate moving average starting from episode 100
            means = c_e_r_t[100:].unfold(0, 100, 1).mean(1).view(-1)
            # Plot the average line starting from episode 100
            plt.plot(range(100, len(data_env)-99), means.numpy(), label='Cumulative Environment Reward')
            means = c_a_r_t[100:].unfold(0, 100, 1).mean(1).view(-1)
            # Plot the average line starting from episode 100
            plt.plot(range(100, len(data_aux)-99), means.numpy(), label='Cumulative Auxiliary Reward')
    plt.legend()
    plt.savefig(f"DQN_Comp_Cumul_Reward_Hyperparameters_{hyperparameters}.png", dpi=300)
    plt.close()

    

    
""""1.2. Define replay buffer and deep Q network"""

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayBuffer(object):

    def __init__(self, capacity):
        self.buffer = deque([], maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

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
        self.cumulative_auxiliary_reward_per_episode = []
        self.agent_performance = []
        self.testing_durations = []

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
        
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        expected_state_action_values = expected_state_action_values.float()

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        return loss.item()

    def update(self, num_episodes):
        for episode in tqdm(range(num_episodes), desc="Episodes"):
            state, info = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            cumulative_reward_t = 0
            environment_reward_t = 0
            auxiliary_reward_t = 0
            running_loss = []
            for t in count():
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = env.step(action.item())
                environment_reward_t += reward
                aux_reward = 3*(observation[0] + 0.5)**2
                auxiliary_reward_t += aux_reward
                reward = reward + aux_reward
                cumulative_reward_t += reward
                reward = torch.tensor([reward], device=device)
                done = terminated or truncated
                
                if terminated:
                    next_state = None
                    self.agent_performance.append(1)
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                if truncated:
                    self.agent_performance.append(0)
                    
                self.memory.push(state, action, next_state, reward)
                state = next_state
                
                loss_value = self.optimize_model()
                
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    self.episode_durations.append(t + 1)
                    break
                    
                running_loss.append(loss_value)
            
            filtered_running_loss = [loss for loss in running_loss if loss is not None]
            if filtered_running_loss:
                self.loss_per_episode.append(np.mean(filtered_running_loss))

            self.cumulative_reward_per_episode.append(cumulative_reward_t)
            self.cumulative_environment_reward_per_episode.append(environment_reward_t)
            self.cumulative_auxiliary_reward_per_episode.append(auxiliary_reward_t)

        torch.save(self.policy_net.state_dict(), 'dqn_policynet.pth')

        
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

"""" 3. Run Model """

def train_agent_with_hyperparameters(BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR, num_episodes=1000):
    n_actions = env.action_space.n
    n_observations = env.observation_space.shape[0]
    agent = DQNAgent(BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR, n_actions, n_observations)
    agent.update(num_episodes)
    hyperparameters = f"BATCH_SIZE={BATCH_SIZE}_GAMMA={GAMMA}_EPS_START={EPS_START}_EPS_END={EPS_END}_EPS_DECAY={EPS_DECAY}_TAU={TAU}_LR={LR}"
    
    # Save the list to a file
    with open('episode_durations.json', 'w') as f:
        json.dump(agent.episode_durations, f)
    with open('cumulative_environment_reward_per_episode.json', 'w') as f:
        json.dump(agent.cumulative_environment_reward_per_episode, f)

    plot_durations(agent.episode_durations, hyperparameters)
    plot_curve(agent.loss_per_episode, 'Loss Curve', 'Training Step', 'Loss', hyperparameters)
    plot_cumul_reward(agent.cumulative_reward_per_episode, 'Episode', 'Cumulative Reward', hyperparameters)
    plot_comp_cumul_reward(agent.cumulative_reward_per_episode, agent.cumulative_environment_reward_per_episode, agent.cumulative_auxiliary_reward_per_episode, hyperparameters)
    cumulative_successes = cumulative_sum(agent.agent_performance)
    plot_curve(cumulative_successes, 'Cumulative Number of Successes', 'Episodes', 'Successes', hyperparameters)

def evaluate_agent(modelfile, seedfile, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR, num_episodes=1000):
    n_actions = env.action_space.n
    n_observations = env.observation_space.shape[0]
    agent = DQNAgent(BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR, n_actions, n_observations)
    agent.evaluate(modelfile, seedfile, num_episodes)
    hyperparameters = f"BATCH_SIZE={BATCH_SIZE}_GAMMA={GAMMA}_EPS_START={EPS_START}_EPS_END={EPS_END}_EPS_DECAY={EPS_DECAY}_TAU={TAU}_LR={LR}"
    
    with open('dqn_heur_testing_durations.json', 'w') as f:
        json.dump(agent.testing_durations, f)

    plot_evaluation(agent.testing_durations, hyperparameters)
    
def hyperparameter_tuning():
    BATCH_SIZE = 64
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.1
    EPS_DECAY = 100000
    TAU = 0.0005 # maybe also 0.005
    LR = 1e-4

    for BATCH_SIZE in [32, 64, 128]:
        train_agent_with_hyperparameters(BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR)
    for GAMMA in [0.9, 0.95, 0.999, 0.99]:
        train_agent_with_hyperparameters(BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR)
    for EPS_START in [1.0, 0.8, 0.9]:
        train_agent_with_hyperparameters(BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR)
    for EPS_END in [0.01, 0.05, 0.1]:
        train_agent_with_hyperparameters(BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR)
    for EPS_DECAY in [10000, 1000000, 100000]:
        train_agent_with_hyperparameters(BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR)
    for TAU in [0.00005, 0.005, 0.0005]:
        train_agent_with_hyperparameters(BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR)
    for LR in [0.001, 0.0005, 0.0001]:
        train_agent_with_hyperparameters(BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR)


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')

    # set up matplotlib
    is_ipython = 'inline' in matplotlib.get_backend()  # checks if one is running IPhyton environment like jupyter notebook
    if is_ipython:
        from IPython import display

    plt.ion()  # interactive mode on, allows automatic plots when data is updated (whithout calling plt.show every time)
    # Run the hyperparameter tuning
    #hyperparameter_tuning()
    # Run the agent training
    train_agent_with_hyperparameters(64, 0.99, 0.9, 0.1, 100000, 0.0001, 1e-4, 3000)
    evaluate_agent('dqn_policynet.pth', 'seeds.json', 64, 0.99, 0.9, 0.1, 100000, 0.0001, 1e-4, 1000)