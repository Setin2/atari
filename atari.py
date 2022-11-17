import gym
import random
import math
import torch
import torchbnn
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

"""
all observation spaces have shape: Box(0, 255, (210, 160, 3), uint8)

state[0] is a list of length 210
state[0][0] is a list of length 160
state[0][0][0] is a list of length 3

so basically state[0] is the actual state we need to pass to the neural net

and state[1] will return something like {'lives': 5, 'episode_frame_number': 0, 'frame_number': 0}
"""

# project description says to use v0
# doing so gives an out of date warning so we might have to ask

# all action spaces are discrete
# action_space = 4
env_breakout = gym.make('Breakout-v0')
# action_space = 6
env_space_invaders = gym.make('SpaceInvaders-v4')
# action_space = 18
env_tennis = gym.make('Tennis-v4')

EPS_DECAY = 0.99  # e-greedy threshold decay
BATCH_SIZE = 256  # Q-learning batch size

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class BayesianNet(torch.nn.Module):
  def __init__(self, num_actions):
    super(BayesianNet, self).__init__()
    self.input = torchbnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=84*84, out_features=64)
    self.output = torchbnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=64, out_features=num_actions)

  def forward(self, state):
    action = torch.relu(self.input(state))
    action = self.output(action)  # no softmax: CrossEntropyLoss() 
    return action

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# non-bayesian implementation of CNN architecture. Only for comparison.
class CNN(torch.nn.Module):
  def __init__(self, num_actions):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4) # in: 1 x 84 x 84, out: 32 x 20 x 20 
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2) # in: 32 x 20 x 20, out: 64 x 9 x 9
    self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1) # in: 64 x 9 x 9, out: 64 x 7 x 7
    self.dense1 = nn.Linear(in_features=64*7*7, out_features=512) # in: 64 * 7 * 7, out: 512
    self.dense2 = nn.Linear(in_features=512, out_features=num_actions) # in: 512, out: num_actions

  def forward(self, state):
    # feature extractor
    features = torch.relu(self.conv1(state))
    features = torch.relu(self.conv2(features))
    features = torch.relu(self.conv3(features))
    features = features.flatten(start_dim=1)

    # classifier
    action = torch.relu(self.dense1(features))
    action = self.dense2(action)

    return action

class SARSA():
  def __init__(self, env, num_actions, learning_rate=1e-3, discount_factor=0.99):
    self.env = env
    self.policy = CNN(num_actions=num_actions).to(device)
    self.optimizer = optim.Adam(self.policy.parameters(), learning_rate)
    self.memory = ReplayMemory(400000)

    self.num_actions = num_actions
    self.discount_factor = discount_factor
    self.steps_done = 0
    self.eps_threshold = 1
    self.max_steps_per_episode = 10000
    super().__init__()
  
  def select_action(self, state):
    sample = random.random()
    self.steps_done += 1

    if sample > self.eps_threshold:
      q_values = self.policy(state)
      return int(q_values.argmax())
    else:
      return random.randrange(self.num_actions)

  def train(self, state):
    state = self.reshape_image(state).to(device)
    total_reward = 0
    for step in range(self.max_steps_per_episode):
      action = self.select_action(state)
      next_state, reward, done, _, _ = self.env.step(action)
      total_reward += reward
      next_state = self.reshape_image(next_state).to(device)
      next_action = self.select_action(next_state)
      self.memory.push((state, action, reward, next_state, next_action, done))
      loss = self.learn_SARSA()
      if done:
        break
      state = next_state

    self.eps_threshold *= EPS_DECAY
    print("epsilon:", np.round(self.eps_threshold,3), ", reward:", np.round(total_reward,3), ", steps:", step, "loss:", np.round(loss,8))

    return total_reward, loss
  
  def reshape_image(self, image):
    if len(image[0]) == 210:
        image = image[0]
    print(image)
    image_data = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)
    image_data[image_data > 0] = 255
    image_data = np.reshape(image_data,(1, 1, 84, 84))
    image_tensor = image_data.astype(np.float32)
    image_tensor = torch.from_numpy(image_tensor)
    return image_tensor

  """def reshape_image(self, image):
    image = cv2.resize(image, dsize=(84, 84), interpolation=cv2.INTER_CUBIC) # change resoluation to 84x84
    image = image[:, :, :1] # keep only on image channel
    return image"""

  def learn_SARSA(self):
    if len(self.memory) > BATCH_SIZE:
      transitions = self.memory.sample(BATCH_SIZE)
      batch_state, batch_action, batch_reward, batch_next_state, batch_next_action, batch_done = zip(*transitions)

      batch_state = torch.vstack(batch_state)
      batch_action = torch.tensor(batch_action).to(device)
      batch_next_state = torch.vstack(batch_next_state).to(device)
      batch_next_action = torch.tensor(batch_next_action).to(device)     
      batch_reward = torch.tensor(batch_reward).to(device)
      batch_done = torch.tensor(batch_done).to(device)

      current_q_values = self.policy(batch_state).gather(1, batch_action.unsqueeze(1)) # Q(St, At)
      next_q_values = self.policy(batch_next_state).gather(1, torch.tensor(batch_next_action).unsqueeze(1)) # Q(St+1, At+1)
      expected_q_values = torch.tensor(batch_reward).unsqueeze(1) + (self.discount_factor * next_q_values)

      loss = F.mse_loss(current_q_values, expected_q_values.view(-1, 1))
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      return loss.cpu().detach().numpy()
    return 0

EPISODES = 10
LEARNING_RATE = 0.00025
env = gym.make('Pong-v0')
print(env.action_space.n)
agent = SARSA(env, env.action_space.n, learning_rate=LEARNING_RATE)
rewards = []
losses = []
ln = list(range(0, EPISODES))
for e in range(EPISODES):
  print("episode:", e)
  state = env.reset()
  reward, loss = agent.train(state)
  rewards.append(reward)
  losses.append(loss)

import matplotlib.pyplot as plt
plt.plot(np.convolve(rewards, np.ones(20) / 20, mode="valid"))

plt.plot(np.convolve(losses, np.ones(5) / 5, mode="valid"))
