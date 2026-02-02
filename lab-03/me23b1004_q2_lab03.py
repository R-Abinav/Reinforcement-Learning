import numpy as np
import gymnasium as gym
from gymnasium import Env, spaces
import matplotlib.pyplot as plt

class CustomMountainCarEnv(Env):
    def __init__(self, min_position=-1.2, max_position=0.6, max_speed=0.07, 
                 goal_position=0.5, force=0.0015, gravity=0.0025, min_steps=10, max_steps=200):
        super(CustomMountainCarEnv, self).__init__()
        
        # environment parameters
        self.min_position = min_position
        self.max_position = max_position
        self.max_speed = max_speed
        self.goal_position = goal_position
        self.force = force
        self.gravity = gravity
        self.min_steps = min_steps
        self.max_steps = max_steps
        
        # action space: 0 - push left, 1 - no action, 2 - push right
        self.action_space = spaces.Discrete(3)
        
        # observation space: [position, velocity]
        self.observation_space = spaces.Box(
            low=np.array([self.min_position, -self.max_speed], dtype=np.float32),
            high=np.array([self.max_position, self.max_speed], dtype=np.float32),
            dtype=np.float32
        )
        
        # state variables
        self.state = None
        self.current_steps = 0
        
    def reset(self):
        # initialize car at bottom of valley with zero velocity
        self.state = np.array([-0.5, 0.0], dtype=np.float32)
        self.current_steps = 0
        return self.state
    
    def step(self, action):
        position, velocity = self.state
        
        # apply transition dynamics
        # v_t+1 = v_t + (a_t - 1) * F + g * cos(3 * x_t)
        velocity += (action - 1) * self.force + self.gravity * np.cos(3 * position)
        
        # clip velocity to allowed range
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        
        # update position
        position += velocity
        
        # clip position to allowed range
        position = np.clip(position, self.min_position, self.max_position)
        
        # if car hits left boundary, reset velocity to zero
        if position == self.min_position and velocity < 0:
            velocity = 0.0
        
        # update state
        self.state = np.array([position, velocity], dtype=np.float32)
        self.current_steps += 1
        
        # reward function: -1 for each step
        reward = -1
        
        # check termination conditions
        done = False
        if position >= self.goal_position and self.current_steps >= self.min_steps:
            done = True
        elif self.current_steps >= self.max_steps:
            done = True
        
        return self.state, reward, done, {}
    
    def render(self):
        position, velocity = self.state
        print(f"position: {position:.4f}, velocity: {velocity:.4f}, steps: {self.current_steps}")

# create environment
env = CustomMountainCarEnv(min_steps=10, max_steps=200)

# run one episode with random policy
episodes = 1

for epi in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0
    
    print(f"\nepisode {epi + 1} started")
    print(f"initial state - position: {state[0]:.4f}, velocity: {state[1]:.4f}")
    
    while not done:
        # random action
        action = env.action_space.sample()
        
        # take step
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
        # action names
        action_names = ["push left", "no action", "push right"]
        
        # display step info
        print(f"\naction: {action_names[action]}")
        env.render()
        print(f"reward: {reward}")
        
        # check if goal reached
        if done and next_state[0] >= env.goal_position:
            print("\ngoal reached!")
        elif done and env.current_steps >= env.max_steps:
            print("\nmax steps reached")
    
    print(f"\ntotal reward for episode {epi + 1}: {total_reward}")
    print(f"final position: {next_state[0]:.4f}, final velocity: {next_state[1]:.4f}")