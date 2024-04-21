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

env = gym.make('CartPole-v1')
observation, info = env.reset(seed=42)

action = env.action_space.sample()  # this is where you would insert your policy
observation, reward, terminated, truncated, info = env.step(action)

BATCH_SIZE = 512
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01 #changed from 0.05
EPS_DECAY = 500 #1000
TAU = 0.005
LR = 2e-4 #1e-4

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReversiModel:
    def __init__(self):
        pass
    
    def predict(self):
        pass

class CurrentBoardEnviroment:
    def __init__(self):
        pass
    
    def getState(self):
        pass
    
class WeightedBoardEnviroment:
    def __init__(self):
        pass
    def getState(self):
        pass
        
class PolicyNet(nn.Module): #Actor
    def __init__(self):
        pass 
    
    def forward(self):
        pass
class ValueNet(nn.Module): #Critic
    def __init__(self):
        pass 
    
    def forward(self):
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
        
        
def train():
    pass
    
def test():
    pass
    
        
    
