import gym
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

# Initialize environment
environment = gym.make('FrozenLake-v1', is_slippery = False, render_mode = 'ansi')  # render_mode: for visualization / is_slippery: parameter that assumes that the model moves on a slippery surface in the environment.
environment.reset()

nb_states = environment.observation_space.n  # number of states
nb_actions = environment.action_space.n  # number of actions
qtable = np.zeros((nb_states, nb_actions))  # initialize Q-table

print('Q-table: ')
print(qtable) # brain of the model

action = environment.action_space.sample()

new_state, reward, done, info, _ = environment.step(action)

# Defining the number of episodes
episodes = 1000
alpha = 0.5  # learning rate
gamma = 0.9  # discount factor

outcomes = []

# Training
for _ in tqdm(range(episodes)):
  state, _ = environment.reset()
  done = False  # success status of model
  outcomes.append('Failure')

  while not done:  # Move within the state until the model is successful (select action and apply)
    # Choose action based on Q-table
    if np.max(qtable[state]) > 0:
      action = np.argmax(qtable[state])
    else:
      action = environment.action_space.sample()  # Random action

    # Take action and get the next state, reward, done flag
    new_state, reward, done, info, _ = environment.step(action)

    # Update Q-table using Q-learning formula
    qtable[state, action] = qtable[state, action] + alpha * (reward + gamma * np.max(qtable[new_state]) - qtable[state, action])

    state = new_state

    if reward:
      outcomes[-1] = 'Success'

print('Q-Table After Training: ')
print(qtable)

plt.bar(range(episodes), outcomes)
plt.show()

# Test
nb_success = 0

for _ in tqdm(range(episodes)):
  state, _ = environment.reset()
  done = False

  while not done:
    if np.max(qtable[state]) > 0:
      action = np.argmax(qtable[state])
    else:
      action = environment.action_space.sample()

    new_state, reward, done, info, _ = environment.step(action)

    state = new_state

    nb_success += reward

print('Success rate: ', 100*nb_success / episodes)