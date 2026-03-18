#=========================
#Soft Actor Critic
#Environment: Pendulum-v1
#=========================
import gymnasium as gym
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import time
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from collections import deque
from torch.distributions.normal import Normal

# =========================
# SEED CONTROL 
# =========================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

# =========================
# Stocastic Policy
# =========================

class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_dim, high, low):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(state_size, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean = nn.Linear(hidden_dim, action_size)
        self.log_std = nn.Linear(hidden_dim, action_size)
        
        self.high = torch.tensor(high).to(device)
        self.low = torch.tensor(low).to(device)
        
        self.apply(weights_init_)
        
        self.register_buffer("action_scale", torch.tensor((high - low) / 2., dtype = torch.float32))
        self.register_buffer("action_bias", torch.tensor((high + low) / 2., dtype = torch.float32))
    
    def forward(self, state):
        log_std_min = -20
        log_std_max = 2
        x = state
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        m = self.mean(x)
        s = self.log_std(x)
        s = torch.clamp(s, min = log_std_min, max = log_std_max)
        return m, s
    
    def sample(self, state):
        noise  = 1e-6
        m, s   = self.forward(state) 
        std    = s.exp()
        normal = Normal(m, std)
        
        a = normal.rsample()
        tanh = torch.tanh(a)
        action = tanh * self.action_scale + self.action_bias
        
        logp = normal.log_prob(a)

        logp -= torch.log(self.action_scale * (1.0 - tanh.pow(2)) + 1e-6)
        logp = logp.sum(1, keepdim=True)
        
        return action, logp

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()

        self.linear1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.linear4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_) 

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)

        q1 = F.relu(self.linear1(state_action))
        q1 = F.relu(self.linear2(q1))
        q1 = self.linear3(q1)

        q2 = F.relu(self.linear4(state_action))
        q2 = F.relu(self.linear5(q2))
        q2 = self.linear6(q2)
        return q1, q2
    
class ReplayMemory:
    def __init__(self, state_dim, action_dim, memory_capacity, batch_size):
        self.capacity = memory_capacity
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((memory_capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((memory_capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((memory_capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((memory_capacity, state_dim), dtype=np.float32)
        self.mask = np.zeros((memory_capacity, 1), dtype=np.float32)

    def push(self, state, action, reward, next_state, mask):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.mask[self.ptr] = mask

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self):
        ind = np.random.randint(0, self.size, size=self.batch_size)
        return (
        self.state[ind],
        self.action[ind],
        self.reward[ind],
        self.next_state[ind],
        self.mask[ind]
               )

    def __len__(self):
        return self.size

class Sac_agent:
    def __init__(self, state_size, action_size, hidden_dim, high, low, memory_capacity, batch_size, gamma, tau, num_updates, policy_freq, alpha):

        self.actor = Actor(state_size, action_size, hidden_dim, high, low).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        
        self.critic = Critic(state_size, action_size, hidden_dim).to(device)   
        
        self.critic_target = Critic(state_size, action_size, hidden_dim).to(device)        
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-4)
        
        self.hard_update(self.critic_target, self.critic)
                
        self.state_size = state_size
        self.action_size = action_size
        
        self.memory = ReplayMemory(state_size, action_size, memory_capacity, batch_size)
        self.gamma = gamma
        self.tau = tau
        self.num_updates = num_updates
        self.update_count = 0
        self.policy_freq=policy_freq
        
        self.target_entropy = -float(self.action_size)
        self.log_alpha = torch.zeros(1, requires_grad=True, device = device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=1e-4)
        self.alpha = self.log_alpha.exp().detach()
        
    def hard_update(self, target, network):
        for target_param, param in zip(target.parameters(), network.parameters()):
            target_param.data.copy_(param.data)
            
    def soft_update(self, target, network):
        for target_param, param in zip(target.parameters(), network.parameters()):
            target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)
            
    def learn(self, batch):
              
        state, action, reward, next_state, mask = batch

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor (next_state).to(device)
        reward = torch.FloatTensor(reward).to(device)
        action = torch.FloatTensor(action).to(device)
        done = torch.FloatTensor(mask).to(device)
                         
        with torch.no_grad():
            act_next, logp_next = self.actor.sample(next_state)
            Q1_target, Q2_target = self.critic_target(next_state, act_next)
            min_Q_target = torch.min(Q1_target, Q2_target)
            Q_target_main = reward + self.gamma * (1 - done) * (min_Q_target - self.alpha.detach() * logp_next)
            
        critic_1, critic_2 = self.critic(state, action)
        critic_loss1 = 0.5*F.mse_loss(critic_1, Q_target_main) 
        critic_loss2 = 0.5*F.mse_loss(critic_2, Q_target_main) 
        total_critic_loss = critic_loss1 + critic_loss2 

        self.critic_optimizer.zero_grad()
        total_critic_loss.backward() 
        self.critic_optimizer.step() 

        act_pi, log_pi = self.actor.sample(state)
        Q1_pi, Q2_pi = self.critic(state, act_pi)
        min_Q_pi = torch.min(Q1_pi, Q2_pi)
        actor_loss =-(min_Q_pi-self.alpha*log_pi ).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
            
        alpha_loss = (-self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        self.update_count += 1
        if (self.update_count % self.policy_freq == 0):
            self.soft_update(self.critic_target, self.critic) 
        
    def act(self, state):
        state =  torch.tensor(state).unsqueeze(0).to(device).float()
        action, logp = self.actor.sample(state)
        return action.cpu().data.numpy()[0]
    
    def step(self):
        self.learn(self.memory.sample())

    def save(self):
        torch.save(self.actor.state_dict(), "pen_actor.pkl")
        torch.save(self.critic.state_dict(), "pen_critic.pkl")
        
def sac(episodes):
    agent = Sac_agent(state_size = state_size, action_size = action_size, hidden_dim = hidden_dim, high = high, low = low, 
                  memory_capacity = memory_capacity, batch_size = batch_size, gamma = gamma, tau = tau, 
                  num_updates = num_updates, policy_freq =policy_freq, alpha = entropy_coef)
    time_start = time.time()
    reward_list = []
    avg_score_deque = deque(maxlen = 100)
    avg_scores_list = []
    mean_reward = -20000
    for i in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        episode_steps = 0
        while not done:
            episode_steps+=1 
            if i < 10:
                action = env.action_space.sample()
            else:
                action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            done = terminated or truncated
            done_flag = float(terminated)

            agent.memory.push(state, action, reward, next_state, done_flag)

            if (len(agent.memory) >= agent.memory.batch_size): 
                agent.step()

            total_reward += reward
            state = next_state
            #env.render()

        print(f"episode: {i+1}, steps:{episode_steps}, current reward: {total_reward}")
        reward_list.append(total_reward)
        avg_score_deque.append(total_reward)
        mean = np.mean(avg_score_deque)
        avg_scores_list.append(mean)
        
    agent.save()
    print(f"episode: {i+1}, steps:{episode_steps}, current reward: {total_reward}, max reward: {np.max(reward_list)}")
           
    return reward_list, avg_scores_list

# =========================
# Environment
# =========================

set_seed(0)
env = gym.make("Pendulum-v1")

action_size = env.action_space.shape[0]
print(f'size of each action = {action_size}')

state_size = env.observation_space.shape[0]
print(f'size of state = {state_size}')

low = env.action_space.low
high = env.action_space.high

print(f'low of each action = {low}')
print(f'high of each action = {high}')

batch_size=256 
memory_capacity = 100000 
gamma = 0.99            
tau = 0.005               
num_of_train_episodes = 1500
num_updates = 1 
policy_freq= 2 
num_of_test_episodes=200
hidden_dim=256
entropy_coef = 0.2 

# =========================
# Traning agent
# =========================

reward, avg_reward = sac(num_of_train_episodes)

new_env = gym.make("Pendulum-v1")
best_actor = Actor(state_size, action_size, hidden_dim = hidden_dim, high = high, low = low)
best_actor.load_state_dict(torch.load("pen_actor.pkl"))        
best_actor.to(device) 
reward_test = []
for i in range(num_of_test_episodes):
    state, _ = new_env.reset()
    local_reward = 0
    done = False
    while not done:
        state =  torch.FloatTensor(state).unsqueeze(0).to(device)
        action, _ = best_actor.sample(state)
        action = action.cpu().data.numpy()[0]
        next_state, r, terminated, truncated, _ = new_env.step(action)
        done = terminated or truncated

        local_reward += r
        state = next_state
    reward_test.append(local_reward)

x = np.array(range(len(reward_test)))
m = np.mean(reward_test)

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=reward_test, name='test reward', line=dict(color="green", width=1)))

fig.add_trace(go.Scatter(x=x, y=[m]*len(reward_test), name='average reward', line=dict(color="red", width=1)))
    
fig.update_layout(title="SAC", xaxis_title= "test", yaxis_title= "reward")
fig.show()

print("Average Reward:", m)