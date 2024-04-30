import gymnasium as gym
import math
import random
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#Initialize the Environment and make an Initial Random Step
env = gym.make('CartPole-v1')
observation, info = env.reset(seed=42)
action = env.action_space.sample() 
observation, reward, terminated, truncated, info = env.step(action)

#Constants
BATCH_SIZE = 128    #Numbers of Samples fed into the neural network during training at once
GAMMA = 0.99        #The decaying factor in the bellman function. Still remember the accumulated *discounted* return?
TAU = 0.005         #Update rate of the duplicate network
LR = 1e-4           #Learning rate of your Q - network

###CREATED EPSILON 
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000


# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)
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
    #######################LOOK INTO READ HINTS #################################
    ########IM USING N_OBSERVATIONS BUT WHAT ABOUT N_ACTIONS AND IT SAYS LAYERS
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        #TODO : Define the layer of your Q network here. Think about the shapes when you define it.
        #What is the shape of the input? What should be the shape of the output?
        
        self.fc1 = nn.Linear(n_observations, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)
      
    def forward(self, x):
        #TODO : Define how the network should process your input to produce an output
        #Should return a tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        return self.fc3(x)
    #############################################################
    #############
   # https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
    #https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
    ##https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    #############
      
#Creating to instances of the Q-network.
#Policy net is trained online directly by loss function
#Target network updates slower and provides a more stable target
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

def select_action(state):
    #TODO : Implement an epsilon-greedy policy that
    #Picks a random action with a small posibility 
    #and acts according to the Q values otherwise
    global steps_done
    ####### THERES A REASON MATH IS IMPORTED USE IT 
    ##ISUE is that the epsilon should degrade and change overtime 
    epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * steps_done / EPS_DECAY)
    
    steps_done += 1
    
    if random.random() < epsilon: #if less than epsilon
        ##SIMILAR TO LINE ABOVE; SELECTS A RANDOM ACTION SAMPLE
       #action = env.action_space.sample() #similar to line above  #needs to perform an action 
        #action = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
        #return action
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
        #return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
    else:
        ## DO SOMETHING HERE
        ## policy_net = DQN(n_observations, n_actions).to(device)
        ## state: represents 1D array of actions like cart pole, cart velocity, pole angle, etc..
        ## Q_Value: Finds the most optimal course of action to take

        with torch.no_grad():
            # Greedy action
            #dq_values = policy_net(state)
            #action = torch.argmax(dq_values).unsqueeze(0)
            return policy_net(state).max(1).indices.view(1, 1)

    #return action



    #  if random.random() > epsilon_greedy_policy: #if less than epsilon
        
        
    #     with torch.no_grad():
    #             # Greedy action
    #         return policy_net(state).max(1).indices.view(1,1)
        
    # else:
    #     return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long) 



episode_durations = []
steps_done = 0


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    #Obtain indices for all non-final states
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                        batch.next_state)), device=device, dtype=torch.bool)
    #The next states of all the non-final states
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    #Gather output of the Q-network
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    #######################
    #ADDED HERE
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    ######################
    with torch.no_grad():
        #TODO : Calculate the observed the Q value (Q_observed = immediate reward + gamma * max(Q(s_t+1)))
        #HINT : Use the target net for estimation of the next state

    #TODO : Pick an appropiate loss function and calculate the loss
    #TODO : Name your calculated loss "loss"
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
      # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    
#####################VERIFY THIS #################################
 # Calculate the expected state-action values using the target network
    #     next_state_values = torch.zeros(BATCH_SIZE, device=device)
    #     next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    #     expected_state_action_values = reward_batch + GAMMA * next_state_values.unsqueeze(1)

    # # Compute the Huber loss between predicted and expected Q-values
    # loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
############################################################


    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


if torch.cuda.is_available():
    num_episodes = 500
else:
    num_episodes = 50

for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    total_reward = 0
    num_steps = 0
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

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

env.close()

TestEnv = gym.make("CartPole-v1", render_mode="human")
observation, info = TestEnv.reset(seed=42)

end_count = 0
for _ in range(1000):
    state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    action = select_action(state) # this is where you would insert your policy
    observation, reward, terminated, truncated, _ = TestEnv.step(action.item())

    if terminated or truncated:
        end_count += 1
        observation, info = TestEnv.reset()

TestEnv.close()
print(end_count)