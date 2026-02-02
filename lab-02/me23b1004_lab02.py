#import 
import numpy as np
import gymnasium as gym
from gym import Env, spaces

#custom cliff walker
class CustomCliffWalker(Env):
  def __init__(self, rows=4, cols=12):
    super(CustomCliffWalker, self).__init__()

    #grid dimensions
    self.rows = rows
    self.cols = cols

    #discrete actions
    self.action_space = spaces.Discrete(4)

    #observation space
    self.observation_space = spaces.Discrete(rows * cols)

    #initialize empty grid layout
    self.layout = np.zeros((rows, cols), dtype=int)

    #place goal at bottom-right corner
    self.layout[-1, -1] = 2

    #place cliff cells on bottom row between start and goal
    for c in range(1, self.cols - 1):
      self.layout[-1, c] = 1

    #set starting position at bottom-left corner
    self.start_pos = (self.rows - 1, 0)
    self.state = (0, -1)

  def step(self, action):
    #extract current position
    x, y = self.state

    #process action with boundary constraints
    if action == 0:
      #move left
      y = max(0, y - 1)
    elif action == 1:
      #move down
      x = min(self.rows - 1, x + 1)
    elif action == 2:
      #move right
      y = min(self.cols - 1, y + 1)
    elif action == 3:
      #move up
      x = max(0, x - 1)

    #update agent state
    self.state = (x, y)
    
    #calculate reward: -1 per step, -100 for cliff, 0 for goal
    reward = -1
    if self.layout[x, y] == 1:
      #penalty for falling into cliff
      reward = -100
    elif self.layout[x, y] == 2:
      #no penalty for reaching goal
      reward = 0

    #episode ends when reaching goal or cliff
    done = self.layout[x, y] == 2 or self.layout[x, y] == 1

    #return state, reward, termination flag, and info dict
    return self._get_state_index(), reward, done, {}

  def reset(self):
    #reset agent to starting position
    self.state = self.start_pos
    return self._get_state_index()

  def render(self):
    #create string representation of grid
    grid = np.array(self.layout, dtype=str)
    grid[self.layout == 0] = "."
    grid[self.layout == 1] = "C"
    grid[self.layout == 2] = "G"

    #get current agent position
    x, y = self.state

    #place agent on grid
    grid[x, y] = "A"

    #display grid with proper formatting
    print("\n".join(" ".join(row) for row in grid))

  def _get_state_index(self):
    #encode 2d position (row, col) as single integer
    return self.state[0] * self.cols + self.state[1]

#initialize custom cliff walker environment
env = CustomCliffWalker(rows=4, cols=12)

#run multiple episodes with random policy
episodes = 5
for epi in range(episodes):
  #reset environment to initial state
  state = env.reset()
  done = False
  total_reward = 0

  print(f"\n------Episode: {epi + 1} --------")
  env.render()

  #run episode until termination
  while not done:
    #select random action from action space
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)

    #accumulate total reward
    total_reward += reward 

    #display step information
    action_names = ["Left", "Down", "Right", "Up"]
    print(f"\nAction: {action_names[action]}")
    env.render()
    print(f"Reward: {reward}")

    #check termination reason
    if reward == -100:
      print("Agent fell into the cliff!!")
    elif reward == 0:
      print("Agent reached the goal!!")

  #print episode summary
  print(f"Episode Total Reward: {total_reward}")