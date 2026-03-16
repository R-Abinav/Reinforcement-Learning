import numpy as np
import gymnasium as gym
from gymnasium import Env, spaces

class PendulumEnv(Env):
    def __init__(self):
        super(PendulumEnv, self).__init__()
        
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, -8.0]),
            high=np.array([1.0, 1.0, 8.0]),
            dtype=np.float32
        )
        
        self.g = 10
        self.m = 1
        self.l = 1
        self.dt = 0.05
        self.max_speed = 8
        self.max_steps = 200
        
        self.theta = None
        self.theta_dot = None
        self.current_steps = 0
    
    def reset(self):
        self.theta = np.random.uniform(-np.pi, np.pi)
        self.theta_dot = np.random.uniform(-1, 1)
        self.current_steps = 0
        return self._get_obs()
    
    def _get_obs(self):
        return np.array([np.cos(self.theta), np.sin(self.theta), self.theta_dot], dtype=np.float32)
    
    def _normalize_angle(self, angle):
        return ((angle + np.pi) % (2 * np.pi)) - np.pi
    
    def step(self, action):
        u = np.clip(action[0], -2.0, 2.0)
        
        self.theta_dot = self.theta_dot + (
            (3 * self.g) / (2 * self.l) * np.sin(self.theta) + 
            (3 / (self.m * self.l**2)) * u
        ) * self.dt
        
        self.theta_dot = np.clip(self.theta_dot, -self.max_speed, self.max_speed)
        self.theta = self._normalize_angle(self.theta + self.theta_dot * self.dt)
        
        cost = self.theta**2 + 0.1 * (self.theta_dot**2) + 0.001 * (u**2)
        reward = -cost
        
        self.current_steps += 1
        done = self.current_steps >= self.max_steps or (abs(self.theta) < 0.01 and abs(self.theta_dot) < 0.01)
        
        return self._get_obs(), reward, done, {}
    
    def render(self, action, reward):
        print(f"step: {self.current_steps}")
        print(f"action: {action:.2f}")
        print(f"theta: {self.theta:.2f}")
        print(f"angular velocity: {self.theta_dot:.2f}")
        print(f"reward: {reward:.2f}")


env = PendulumEnv()

print("running random policy for 5 episodes...")

for episode in range(5):
    obs = env.reset()
    done = False
    total_reward = 0
    
    print(f"\nepisode {episode + 1}")
    
    while not done:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        env.render(action[0], reward)
    
    print(f"episode {episode + 1} total reward: {total_reward:.2f}")