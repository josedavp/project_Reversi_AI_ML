import math
import random
import numpy as np
import socket, pickle
from reversi import reversi
from collections import namedtuple, deque
from itertools import count
#from greedy_player import greedy_player

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

Train = True # Boolean flag for training or tetsing code

BATCH_SIZE = 512 # Number of transitions sampled in each optimization step
GAMMA = 0.99 # Discount factor for future rewards
EPS_START = 0.9 # Exploraton rate range
EPS_END = 0.05
EPS_DECAY = 1000 #number of steps for exploration rate decay
TAU = 0.005 # soft update parameter for target network
LR = 1e-4 # learning rate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class GreedyPlayer:
#     def greedyPlayer(self):
#         x = -1
#         y = -1
#         max = 0
#         game.board = board
#         for i in range(8):
#             for j in range(8):
#                 cur = game.step(i, j, turn, False)
#                 if cur > max:
#                     max = cur
#                     x, y = i, j
#         return x,y

game_socket = socket.socket()
game_socket.connect(('127.0.0.1', 33333))
game = reversi()

class ReversiEnvironment:
    """Sets up the Environment for Reversi Game """
    def __init__(self, board=None): #or redundant?
        self.game = reversi() #.reversi()
    
    def reset(self, board):
        """ Resets board for training session """
        self.game = reversi()#.reversi()
        self.game.board = board  # Reset or set specific board
        observation = self.game.board
        info = {}
        return observation, info
    
    def step(self, action):
        """ Interacts with Reversi game server giving next state and reward"""
        print(f"TURN STEP: {self.game.turn}")
        action = action.view(-1)
        x = action[0].item() # changed from [] only to [][]
        y = action[1].item()
        self.game.step(x, y, self.game.turn, commit=True) #commits the move
        game_socket.send(pickle.dumps([x,y]))

        game = ReversiEnvironment()

        data = game_socket.recv(4096)
        turn, board = pickle.loads(data)

        if turn == 0:
            game_socket.close()
            return
        game.game.board = board
        print(f"BOARD STEP: \n {game.game.board}")
        #observation = self.game.board
        reward = self.calculate_reward(game.game.board, x, y) 
        info = {} # Extra Info
        return game.game.board, reward, False, info 

    def stepTest(self, action):
        ############## original ##########
        x = action[0].item()
        y = action[1].item()
        game_socket.send(pickle.dumps([x,y]))
        #data = game_socket.recv(4096)
        #######################################
        game = reversi()
        #reversi_model = ReversiEnvironment()

        data = game_socket.recv(4096)
        turn, board = pickle.loads(data)
        if turn == 0:
            game_socket.close()
            return
        game.board = board
     
    def predict(self, board):
        """Returns random valid move from available actions"""
        available_actions = self.action_space(board)
        return random.choice(available_actions)
    
    def action_space(self, board): #include board as a parameter instead; for only open spaces
        """ Needed to handle possible actions in game board and checks if its a legal move """
        coordinate_actions = []
        available_actions = 0#[]
        board = self.game.board if board is None else board
        for x in range(8):
            for y in range(8):
                if self.game.step(x, y, self.game.turn, commit=False) > 0:
                    coordinate_actions.append((x, y))
                    available_actions += 1
        if available_actions == 0:
            print(f"Action Space Available Actions: {available_actions}")
            x, y = -1, -1
            game_socket.send(pickle.dumps([x,y]))
            #(-1,-1)

        return available_actions, coordinate_actions
    
    def calculate_reward(self, board, x, y):
        """Calculates the reward based on the action taken (placing a piece)"""
        # Get the current player's color
        current_player = self.game.turn
        
         # Convert tensor values to scalar
        x_scalar = x#.item() #
        y_scalar = y#.item() #
        print(f"Calculate reward: x_scalar: {x_scalar}  y_scalar: {y_scalar}")
    
        # Count the total number of pieces for each player
        player_piece_count = self.count_pieces(board, current_player)
        opponent_piece_count = self.count_pieces(board, -current_player)
        
        # Calculate the piece difference
        piece_difference = player_piece_count - opponent_piece_count

        # Reward for placing a piece
        place_piece_reward = 1

        # Reward for corner placement
        corner_reward = 0
        if (x_scalar in [0, 7] and y_scalar in [0, 7]):
            corner_reward = 5  # Adjust weight as needed

        # Combine the rewards with weights
        total_reward = (piece_difference * 0.8) + (place_piece_reward * 0.1) + (corner_reward * 0.05)
        return total_reward

    def count_pieces(self, board, color):
        """Counts the number of pieces of current player color on the board"""
        count = 0
        for row in board:
            for piece in row:
                if piece == color:
                    count += 1
        return count

env = ReversiEnvironment()
observation, info = env.reset(env.game.board)

n_actions = 64
n_observations = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

episode_durations = []
steps_done = 0


def select_action(state):
    """ Selects action move to take """
    global steps_done
    sample = random.random()
   # print(f"sample: {sample}")
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            state = state.view(1,64)
            print(f"Select Action IF State: \n {state}")
            action_index = policy_net(state).max(1)[1].view(1, 1).item()
            print(f"IF Action_Index : {action_index}")
            x, y = env.action_space(state)[1][action_index]
            print(f"IF x: {x} Y: {y}")
            return torch.tensor([[x, y]], device=device, dtype=torch.long)
    else:
        print(f" Select Action ELSE STATE: \n{state}")
        available_actions , options = env.action_space(state)
        if available_actions == 0:
            print(f"IF Select Action Available Coordinates: {available_actions}")
            x, y = -1, -1
            game = ReversiEnvironment()
            game.step(x, y, game.game.turn, commit=True)
            game_socket.send(pickle.dumps([x,y]))
            game_socket.close() #Maybe not here
        else:
            action = random.choice(options)
        return torch.tensor(action, device=device, dtype=torch.long)

def optimize_model():
   if len(memory) < BATCH_SIZE:
      return
   transitions = memory.sample(BATCH_SIZE)
   batch = Transition(*zip(*transitions))
   non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                       batch.next_state)), device=device, dtype=torch.bool)
   non_final_next_states = torch.cat([s for s in batch.next_state
                                             if s is not None])
   state_batch = torch.cat(batch.state)
   action_batch = torch.cat(batch.action)
   reward_batch = torch.cat(batch.reward)

   state_action_values = policy_net(state_batch).gather(1, action_batch)

   next_state_values = torch.zeros(BATCH_SIZE, device=device)
   with torch.no_grad():
      next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
   expected_state_action_values = (next_state_values * GAMMA) + reward_batch

   criterion = nn.SmoothL1Loss()
   loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

   optimizer.zero_grad()
   loss.backward()
   torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
   optimizer.step()
   return loss

if torch.cuda.is_available():
    num_episodes = 500
else:
    num_episodes = 50

if Train:
    writer = SummaryWriter()

    for i_episode in range(num_episodes):
        # Initialize the environment and get it's state
        state, info = env.reset(env.game.board)
        print(f"Train State: \n {state}")
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0
        num_steps = 0
        for t in count():
            action = select_action(state)
            print(f"Train For action: {action}")
            #observation, reward, terminated, truncated, _ = env.step(action.item())
            observation, reward, terminated , _ = env.step(action)
            reward = torch.tensor([reward], device=device)
            done = terminated #or truncated

            if terminated:
                next_state = None
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                break
        if i_episode % 10 == 0:
            print(i_episode)

TestEnv = ReversiEnvironment() #gym.make("CartPole-v1", render_mode="human")
observation, info = TestEnv.reset(TestEnv.game.board) # seed=42)


#########TRYING TO SAVE CHECKPOINT ############
# net = DQN()
# # Additional information
# EPOCH = 5
# PATH = "model.pt"
# LOSS = 0.4

# torch.save({
#             'epoch': EPOCH,
#             'model_state_dict': net.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': LOSS,
#             }, PATH)

# model = Net()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# checkpoint = torch.load(PATH)
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']

# model.eval()
# # - or -
# model.train()
####################





for _ in range(1000):
    state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    action = select_action(state) # this is where you would insert your policy
    observation, reward, terminated, _= TestEnv.step(action)
    if terminated: 
        observation, info = TestEnv.reset(TestEnv.game.board)
print("Ready to Play")