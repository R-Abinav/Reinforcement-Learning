import numpy as np
#import gym
import gymnasium as gym
import random
from gymnasium import Env, spaces
import matplotlib.pyplot as plt

class CustomFrozenlayoutEnv(Env):
    def __init__(self, grid_size=4, slip_probability=0.2, max_steps=50):
        super(CustomFrozenlayoutEnv, self).__init__()
        
        #define grid size
        self.grid_size = grid_size
        self.slip_probability = slip_probability
        self.max_steps = max_steps
        self.current_steps = 0
        
        # action spaces : 0 - left ; 1 - down ; 2 - right ; 3 - up
        self.action_space = spaces.Discrete(4)
        
        # observation space where each state corresponds to a grid cell
        self.observation_space = spaces.Discrete(grid_size * grid_size)
        
        # define layout layout 1: 0 = frozen, 1 = hole, 2 = goal
        self.layout = np.zeros((grid_size, grid_size), dtype=int)
        self.layout[-1, -1] = 2  # Goal
        
        #holes
        for i in range(2, grid_size):
            for j in range(2, grid_size):
                if i==j:  #generate holes on all diags for simplicity
                    self.layout[i, j] = 1
        
        #starting state
        self.start_pos = (0, 0)
        self.state = (0, 0)
    
    def step(self, action):
        x, y = self.state  #read current state
        
        # Check if current position is frozen and apply slip condition BEFORE moving
        if self.layout[x, y] == 0 and random.random() < self.slip_probability:
            action = self.action_space.sample()  # Agent slips - random action
        
        # move agent based on action (original or slipped)
        if action == 0:  # Left
            y = max(0, y - 1)
        elif action == 1:  # Down
            x = min(self.grid_size - 1, x + 1)
        elif action == 2:  # Right
            y = min(self.grid_size - 1, y + 1)
        elif action == 3:  # Up
            x = max(0, x - 1)
        
        #state updation
        self.state = (x, y)
        self.current_steps += 1
        
        #calculate rewards
        reward = -0.1  # Small negative reward for each step
        
        if self.layout[x, y] == 1:  #penalty for falling into hole
            reward = -2
        if self.layout[x, y] == 2:
            reward = 1
        
        # termination check
        done = (self.layout[x, y] == 2 or 
                self.layout[x, y] == 1 or 
                self.current_steps >= self.max_steps)
        
        # return all info
        return self._get_state_index(), reward, done, {}
    
    def reset(self):
        self.state = self.start_pos
        self.current_steps = 0
        return self._get_state_index()
    
    def render(self):
        grid = np.array(self.layout, dtype=str)
        grid[self.layout == 0] = "."
        grid[self.layout == 1] = "H"
        grid[self.layout == 2] = "G"
        x, y = self.state
        grid[x, y] = "A"  # agent
        print("\n".join(" ".join(row) for row in grid))
    
    def _get_state_index(self):
        return self.state[0] * self.grid_size + self.state[1]

env = CustomFrozenlayoutEnv(grid_size=32, slip_probability=0.2, max_steps=50)

episodes = 100
total_rewards = []

for epi in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    print(f"\nEpisode: {epi + 1}\n")
    env.render()
    
    while not done:
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        
        action_names = ["Left", "Down", "Right", "Up"]
        print(f"\nAction: {action_names[action]}")
        env.render()
        print(f"\nReward: {reward}")
        
        #check termination condition
        if reward == -2:
            print("\nAgent fell into the cliff")
        elif reward == 1:
            print("\nAgent achieved the goal!")
        elif env.current_steps >= env.max_steps:
            print("\nEpisode terminated due to steps crossing the limit")
    
    total_rewards.append(total_reward)
    print(f"\nTotal Reward for Episode {epi + 1}: {total_reward}")

#plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, episodes + 1), total_rewards, marker='o', linestyle='-', markersize=3)
plt.xlabel('Episode Number')
plt.ylabel('Total Reward')
plt.title('Total Reward vs Episode Number (Random Policy)')
plt.grid(True)
plt.show()

print(f"\nAverage Total Reward over {episodes} episodes: {np.mean(total_rewards):.2f}")