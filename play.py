"""Watch a trained DQN agent play Atlantis.

Run it like this:
    python play.py --model models/JeanMuhire/exp09_gamma_0.95_batch_64/dqn_model.pth

It loads the saved network, opens a game window and plays with no random moves:
on every screen it presses the button with the highest Q-value. Nothing is
learned here, the network only plays.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from dqn_components import QNetwork, device, make_env

WINDOW_NAME = "DQN agent playing Atlantis (press Q to stop)"
WINDOW_SCALE = 3                # the game screen is small, so show it 3 times bigger
SECONDS_PER_MOVE = 4 / 60       # each move lasts 4 game frames, and the game runs at 60 frames a second


def parse_args():
    parser = argparse.ArgumentParser(description="Watch a trained DQN agent play Atlantis.")
    parser.add_argument("--model", default="dqn_model.pth",
                        help="path to the saved network (dqn_model.pth)")
    parser.add_argument("--episodes", type=int, default=1, help="how many games to play")
    parser.add_argument("--max-episode-steps", type=int, default=10_000,
                        help="stop a game after this many moves, the same limit as in training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-render", action="store_true",
                        help="play without opening a window, only print the scores")
    return parser.parse_args()


def load_network(model_path, n_actions):
    # Build the same CNN that was used in training, then put the saved weights in it.
    network = QNetwork(n_actions).to(device)
    weights = torch.load(model_path, map_location=device)
    network.load_state_dict(weights)
    network.eval()  # evaluation mode: the network only plays, it does not learn
    return network


def choose_best_action(network, state):
    # Greedy choice: always the button with the highest Q-value, never a random one.
    state_tensor = torch.as_tensor(np.asarray(state), device=device).unsqueeze(0)
    with torch.no_grad():
        q_values = network(state_tensor)
    return int(q_values.argmax(dim=1).item())


def show_frame(env, move_started):
    # Draw the current game screen in our window, at normal game speed.
    # Returns False if the viewer pressed Q or closed the window.
    frame = env.render()  # the full-colour game screen
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = cv2.resize(frame, None, fx=WINDOW_SCALE, fy=WINDOW_SCALE,
                       interpolation=cv2.INTER_NEAREST)
    cv2.imshow(WINDOW_NAME, frame)

    time_left = SECONDS_PER_MOVE - (time.time() - move_started)
    key = cv2.waitKey(max(1, int(time_left * 1000)))
    window_closed = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
    return not (key in (ord("q"), ord("Q")) or window_closed)


def play(args):
    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model file not found: {model_path}\n"
                         "Pass the right path with --model.")

    show_window = not args.no_render
    # "rgb_array" lets us grab each game screen and draw it in our own window.
    env = make_env(render_mode="rgb_array" if show_window else None)
    network = load_network(model_path, env.action_space.n)
    print(f"Loaded {model_path}, playing {args.episodes} game(s) on {device}.")

    if show_window:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)  # keep it in front

    scores = []
    stopped = False
    for episode in range(1, args.episodes + 1):
        state, _ = env.reset(seed=args.seed + episode)
        score = 0.0
        steps = 0
        while True:
            move_started = time.time()
            action = choose_best_action(network, state)
            state, reward, terminated, truncated, _ = env.step(action)
            score += reward  # the real game score, not clipped like in training
            steps += 1
            if show_window and not show_frame(env, move_started):
                stopped = True
                break
            if terminated or truncated or steps >= args.max_episode_steps:
                break
        scores.append(score)
        ending = " (stopped early)" if stopped else ""
        print(f"Game {episode}: score {score:,.0f} in {steps:,} moves{ending}")
        if stopped:
            break

    env.close()
    if show_window:
        cv2.destroyAllWindows()
    if len(scores) > 1:
        print(f"Mean score over {len(scores)} games: {np.mean(scores):,.0f}")


if __name__ == "__main__":
    play(parse_args())
