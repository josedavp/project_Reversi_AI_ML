import reversi
import math
import random
from collections import namedtuple, deque
from itertools import count 

import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
import numpy as np

BATCH_SIZE = 512
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01 #changed from 0.05
EPS_DECAY = 500 #1000
TAU = 0.005
LR = 2e-4 #1e-4

# TODO not all these classes will be needed then. 
class ReversiModel: # perhaps combine both current board and weighed board to find best solution? 
    # game is then calculated either here or in reality just by game training model
    # then result is sent back to use
    
    #handle actual reversi content here, add the current and weighted board here
    # optimize it and it will be used and the env with select actions
    # then once handled in ML training and testing output will be returned and send back to main_player.py
    def __init__(self):
        pass
    
    def predict(self, board_state):
        pass

class ReversiEnviroment:
    # Sets up the required reset and step for reversi enviroment
    def __init__(self, board):
        self.state = board
    
    def reset(self, board):
        self.state = board
        return self.game_state
    
    def step(self, action):
        observation = self.game_state
        #reward = #think on this and avid reward abundance perhaps not all the board will be a positive number, also consider instead of manually doing it 
        # instead it could be automatic from the ML learning on its own best area to go to?
        #terminated = T or F
        #info = Extra Info
        #return all data
        return observation # for now 
    
    def action_space(self):
        pass

    #not sure 
    def observation_space(self):
        pass

class CurrentBoardEnviroment: #pass the current board state; therefore naming should be this
    # or perhaps use this class and weighted board for reversimodel work
    def __init__(self):
        pass
    
    def getState(self):
        pass
    
#   TODO
class WeightedBoardEnviroment: #create a weighted multi dimension array/board for optimization
    # similar to the action in last tournament
    def __init__(self):
        pass
    def getState(self):
        pass

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
        
    def __len__(self):
        return len(self.memory)
        
class PolicyNet(nn.Module): #Actor
    def __init__(self, n_observations, n_actions):
        super(PolicyNet, self)._init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions) 
    
    def forward(self):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.softmax(x)
        return x
    
class ValueNet(nn.Module): #Critic
    def __init__(self):
        super(ValueNet,self).__init__()
        self.fc1 = nn.Linear(n_observations, 128) 
        self.fc2 = nn.Linear(128), 128) 
        self.fc3 = nn.Linear(128, 1) # 1 represents single output neuron
    
    def forward(self):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
##########
#TODO Keep here for the moment
env = #gym.make('CartPole-v1')
observation, info = env.reset(seed=42)

action = env.action_space.sample()  # this is where you would insert your policy
observation, reward, terminated, truncated, info = env.step(action)

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
################3

p_net = PolicyNet(n_observations, n_actions).to(device)

optimizer = optim.AdamW(p_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

def select_action(state):
    probs = p_net(state).cpu()
    m = Categorical(probs)
    action = m.sample()
    return action.item(), m.log_prob(action)


        
def train(): #like this for now adjust contents later
    episode_durations = []
    steps_done = 0
    
    if torch.cuda.is_available():
        num_episodes = 500
    else:
        num_episodes = 50

    writer = SummaryWriter()

    device = torch.device('cuda')


    for i_episode in range(num_episodes):
        # Initialize the environment and get it's state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0
        num_steps = 0
        saved_log_probs = []
        rewards = []
        for t in count():
            action, log_prob = select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action)
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Move to the next state
            state = next_state

            rewards.append(reward)
            saved_log_probs.append(log_prob)
            if done:
                episode_durations.append(t + 1)
                break

        returns = deque(maxlen=500) 
        n_steps = len(rewards) 
        for t in range(n_steps)[::-1]:
                disc_return_t = (returns[0] if len(returns)>0 else 0)
                returns.appendleft(GAMMA*disc_return_t + rewards[t])   

        eps = np.finfo(np.float32).eps.item()
        ## eps is the smallest representable float, which is 
        # added to the standard deviation of the returns to avoid numerical instabilities        
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        
        # Line 7:
        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * disc_return)
        policy_loss = torch.cat(policy_loss).sum()
        
        # Line 8: PyTorch prefers gradient descent 
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % 10 == 0:
            print(i_episode)

    env.close()
    
def test():
    TestEnv = #gym.make("CartPole-v1", render_mode="human")
    observation, info = TestEnv.reset(seed=42)

    for _ in range(1000):
        state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        action, _ = select_action(state) # this is where you would insert your policy
        observation, reward, terminated, truncated, _ = TestEnv.step(action)

        if terminated or truncated:
            observation, info = TestEnv.reset()

    TestEnv.close()
    
        
    

