"""The game environment, the CNN and the replay buffer, used by both train.py and play.py."""

import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

gym.register_envs(ale_py)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_env(render_mode=None):
    # frameskip=1 here, because AtariPreprocessing does the frame skipping itself.
    env = gym.make("ALE/Atlantis-v5", frameskip=1, render_mode=render_mode)
    # grayscale + resize to 84x84 + skip 4 frames per action
    env = AtariPreprocessing(env, frame_skip=4, screen_size=84, grayscale_obs=True)
    # stack the last 4 frames so the agent can see motion
    env = FrameStackObservation(env, stack_size=4)
    return env


class QNetwork(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        x = x.float() / 255.0  # scale pixels from 0-255 to 0-1
        return self.fc(self.conv(x))


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.index = 0   # where the next experience goes
        self.size = 0    # how many experiences are stored
        # frames stay uint8 to save memory
        self.states = np.zeros((capacity, 4, 84, 84), dtype=np.uint8)
        self.next_states = np.zeros((capacity, 4, 84, 84), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.dones[self.index] = done
        self.index = (self.index + 1) % self.capacity  # overwrite oldest when full
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.states[idx]).to(device),
            torch.from_numpy(self.actions[idx]).to(device),
            torch.from_numpy(self.rewards[idx]).to(device),
            torch.from_numpy(self.next_states[idx]).to(device),
            torch.from_numpy(self.dones[idx]).to(device),
        )

    def __len__(self):
        return self.size
