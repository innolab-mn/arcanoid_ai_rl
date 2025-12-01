import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from game.arkanoid import ArkanoidGame
from game.settings import SCREEN_HEIGHT, SCREEN_WIDTH

class ArkanoidEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.game = ArkanoidGame(render_mode=render_mode)
        
        # Action space: 0 = Stay, 1 = Left, 2 = Right
        self.action_space = spaces.Discrete(3)
        
        # Observation space: Grayscale 84x84
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(84, 84, 1), 
            dtype=np.uint8
        )

    def _process_obs(self, obs):
        # obs is (H, W, 3)
        # Resize to 84x84
        resized = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        # Convert to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        # Add channel dimension (84, 84, 1)
        return np.expand_dims(gray, axis=-1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.game.reset()
        processed_obs = self._process_obs(obs)
        info = {}
        return processed_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.game.step(action)
        processed_obs = self._process_obs(obs)
        return processed_obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self.game.get_state()
        elif self.render_mode == "human":
            self.game.render()

    def close(self):
        self.game.close()
