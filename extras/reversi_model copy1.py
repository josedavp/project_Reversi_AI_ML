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

Train = True

BATCH_SIZE = 512
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO work on initial template like reversi enviroment then later on think of reversi model for heuristics etc.
class ReversiEnvironment:
    """Sets up the Environment for Reversi Game """
    def __init__(self, board=None): #or redundant?
        self.game = reversi.reversi()
    
    def reset(self, board):
        """ Resets board for training session """
        self.game = reversi.reversi()
        self.game.board = board  # Reset or set specific board
        observation = self.game.board
        info = {}
        return observation, info
    
    def step(self, action): #action, coordinates):
        x = action[0].item()
        y = action[1].item()
        reversi.game_socket.send(pickle.dumps([x,y]))
        data = reversi.game_socket.recv(4096)
        turn, board = pickle.pickle.loads(data)
        return turn, board
     
    def predict(self, board):
        """Returns random valid move"""
        available_actions = self.action_space(board)
        return random.choice(available_actions)
    
    def action_space(self, board): #include board as a parameter instead; for only open spaces
        """ Needed to handle possible actions """
        coordinate_actions = []
        available_actions = 0#[]
        board = self.game.board if board is None else board
        for x in range(8):
            for y in range(8):
                if self.game.step(x, y, self.game.turn, commit=False) > 0:
                    coordinate_actions.append((x, y))
                    available_actions += 1
        return available_actions, coordinate_actions
    
    def calculate_reward(self, board, x, y):
        """Calculates the reward based on the action taken (placing a piece)"""
        # Get the current player's color
        current_player = self.game.turn
        
         # Convert tensor values to scalar
        x_scalar = x#.item() #
        y_scalar = y#.item() #
        print(f"x_scalar: {x_scalar}  y_scalar: {y_scalar}")
    
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
    global steps_done
    sample = random.random()
   # print(f"sample: {sample}")
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            print(f"State : {state}")
            action_index = policy_net(state).max(1)[1].view(1, 1).item()
            x, y = env.action_space(state)[1][action_index]
            return torch.tensor([[x, y]], device=device, dtype=torch.long)
    else:
        print(f"STATE: ")
        _ , options = env.action_space(state)
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
        print(f"state: {state}")
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0
        num_steps = 0
        for t in count():
            action = select_action(state)
            print(f"action: {action}")
            #observation, reward, terminated, truncated, _ = env.step(action.item())
            observation, reward, terminated, _ = env.step(action)
            reward = torch.tensor([reward], device=device)
            done = terminated #or truncated

            if terminated:
                next_state = None
            else:
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

for _ in range(1000):
    state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    action = select_action(state) # this is where you would insert your policy
    observation, reward, terminated, _= TestEnv.step(action)# truncated, _ 

    if terminated: 
        observation, info = TestEnv.reset(TestEnv.game.board)