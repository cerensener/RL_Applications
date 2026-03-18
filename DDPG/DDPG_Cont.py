import gymnasium as gym
import torch
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import math
from collections import deque
import random

#============================
# Networks
#============================

class Critic(nn.Module):

    def __init__(self, obs_dim, action_dim):
        super(Critic, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.linear1 = nn.Linear(self.obs_dim, 1024)
        self.linear2 = nn.Linear(1024 + self.action_dim, 512)
        self.linear3 = nn.Linear(512, 300)
        self.linear4 = nn.Linear(300, 1)

    def forward(self, x, a):
        x = F.relu(self.linear1(x))
        xa_cat = torch.cat([x,a], 1)
        xa = F.relu(self.linear2(xa_cat))
        xa = F.relu(self.linear3(xa))
        qval = self.linear4(xa)

        return qval

#======================
# Deterministic Policy
#======================

class Actor(nn.Module):

    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.linear1 = nn.Linear(self.obs_dim, 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, self.action_dim)

    def forward(self, obs):
        x = F.relu(self.linear1(obs))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        return x  

#=========================
# Ornstein-Ulhenbeck Noise
#=========================

class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)

#==============
# Replay Buffer
#==============

class BasicBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward], dtype = np.float32), next_state, np.array([float(done)], dtype = np.float32))
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (np.array(state_batch), np.array(action_batch), np.array(reward_batch), np.array(next_state_batch), np.array(done_batch))
    
    def __len__(self):
        return len(self.buffer)

#=======================
# DDPG Agent
#=======================

class DDPGAgent:
    
    def __init__(self, env, gamma, tau, buffer_maxlen, critic_learning_rate, actor_learning_rate):
        self.device = torch.device("cpu")
        
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        
        self.gamma = gamma
        self.tau = tau

        self.critic = Critic(self.obs_dim, self.action_dim).to(self.device)
        self.critic_target = Critic(self.obs_dim, self.action_dim).to(self.device)
        
        self.actor = Actor(self.obs_dim, self.action_dim).to(self.device)
        self.actor_target = Actor(self.obs_dim, self.action_dim).to(self.device)
    
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
    
        self.replay_buffer = BasicBuffer(buffer_maxlen)        
        self.noise = OUNoise(self.env.action_space)
        
    def get_action(self, obs): 
        state = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action = self.actor.forward(state)
        action = action.squeeze(0).cpu().detach().numpy()

        return action
    
    def update(self, batch_size):
        states, actions, rewards, next_states, _ = self.replay_buffer.sample(batch_size)

        state_batch, action_batch, reward_batch, next_state_batch, masks = self.replay_buffer.sample(batch_size)
       
    
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device) # Deterministic action selection
        curr_Q = self.critic.forward(state_batch, action_batch) 
        
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        next_actions = self.actor_target.forward(next_state_batch)
        next_Q = self.critic_target.forward(next_state_batch, next_actions.detach())
        
        
        reward_batch = torch.FloatTensor(np.array(reward_batch)).to(self.device)
        masks = torch.FloatTensor(np.array(masks)).unsqueeze(1).to(self.device)
        expected_Q = reward_batch + (1 - masks) * self.gamma * next_Q # self.gamma-->discount factor
        
        q_loss = F.mse_loss(curr_Q, expected_Q.detach())

        self.critic_optimizer.zero_grad()
        q_loss.backward() 
        self.critic_optimizer.step()

        
        
        policy_loss = -self.critic.forward(state_batch, self.actor.forward(state_batch)).mean() # mean of the Q values for all state-action pairs
        
        self.actor_optimizer.zero_grad() # gradients of the actor optimizer to zero for clean optimization
        policy_loss.backward() # backpropagation to compute gradients of "policy_loss"
        self.actor_optimizer.step() # update actor network parameters to based on computed gradients

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))


def mini_batch_train(env, agent, max_episodes, max_steps, batch_size,exploration_noise):
    episode_rewards = []
    
    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            env.render()
            action = agent.get_action(state)
            action= action + exploration_noise.get_action(action)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            episode_reward += reward

            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)
                


            if done or step == max_steps-1:
                episode_rewards.append(episode_reward)
                exploration_noise.reset() # Re-initialize the process when episode ends
                print("Episode " + str(episode) + ": " + str(episode_reward))
                break

            state = next_state
    return episode_rewards

env = gym.make("Pendulum-v1")

max_episodes = 200
max_steps = 1000
batch_size = 32

gamma = 0.99
tau = 1e-2
buffer_maxlen = 100000
critic_lr = 1e-4
actor_lr = 1e-4

exploration_noise = OUNoise(env.action_space)

agent = DDPGAgent(env, gamma, tau, buffer_maxlen, critic_lr, actor_lr)
episode_rewards = mini_batch_train(env, agent, max_episodes, max_steps, batch_size,exploration_noise)

#==================
#Benchmark Graphic
#==================

plt.figure(figsize=(10, 6))

plt.plot(episode_rewards, label='Bölüm (Episode) Ödülü', color='lightblue', alpha=0.8)

window_size = 10
if len(episode_rewards) >= window_size:
    moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size-1, len(episode_rewards)), moving_avg, 
             label=f'{window_size} Bölümlük Hareketli Ortalama', color='red', linewidth=2)

plt.title('DDPG (Pendulum-v1)', fontsize=14, fontweight='bold')
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Reward', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)

plt.tight_layout()
plt.savefig('DDPG_Benchmark.png', dpi=300)
print("Graph saved as 'DDPG_Benchmark.png'!")
plt.show()
