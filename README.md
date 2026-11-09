# DQN on Atari Atlantis

Deep Q-Network (DQN) agent for `ALE/Atlantis-v5`, built from scratch with PyTorch and Gymnasium (no Stable-Baselines3, RLlib, CleanRL or TorchRL).

## Sheilla: Environment, CNN and replay buffer

This notebook contains the building blocks used by `train.py`:

- **Environment (`make_env`)**: Atlantis with grayscale, 84x84 frames, frame skip 4 and a stack of 4 frames. Observation shape is `(4, 84, 84)` (uint8) and there are 4 actions.
- **CNN Q-network (`QNetwork`)**: 3 convolutional layers and 2 fully connected layers that output 4 Q-values. Randomly initialised, no pretrained weights.
- **Replay buffer (`ReplayBuffer`)**: stores `(state, action, reward, next_state, done)` with frames as uint8 and samples random batches.
- **Random baseline**: 10 episodes with random actions, average score **19,540**. A trained agent should beat this.

## Setup

```bash
pip install -r requirements.txt
```

## Files

- `member1_environment_cnn_buffer.ipynb`: environment, CNN, replay buffer and random baseline
- `requirements.txt`: dependencies
