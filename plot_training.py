"""Make training plots from the training_log.csv that train.py saves.

One run folder gives the reward and game length of that run:
    python plot_training.py models/JeanMuhire/exp09_gamma_0.95_batch_64 --output training_plots/JeanMuhire/exp09.png

Several run folders give one chart that compares their scores:
    python plot_training.py models/JeanMuhire/exp01_gamma_0.90 models/JeanMuhire/exp02_gamma_0.95 --output training_plots/JeanMuhire/compare.png
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save the images to files without opening a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

WINDOW = 20  # our team compares runs by the mean score of the last 20 games


def read_log(run_dir):
    """Read a run's training log and return the steps, scores and game lengths."""
    with open(Path(run_dir) / "training_log.csv", newline="") as log_file:
        rows = list(csv.DictReader(log_file))
    steps = np.array([int(row["total_steps"]) for row in rows])
    scores = np.array([float(row["reward"]) for row in rows])
    lengths = np.array([int(row["episode_length"]) for row in rows])
    return steps, scores, lengths


def mean_of_last_games(values):
    """At each game, the mean of the last WINDOW games. It starts once WINDOW games exist."""
    return np.convolve(values, np.ones(WINDOW) / WINDOW, mode="valid")


def plot_one_run(run_dir, output_path, random_score):
    """Two charts side by side: the score and the game length of every game in one run."""
    steps, scores, lengths = read_log(run_dir)
    figure, (score_chart, length_chart) = plt.subplots(1, 2, figsize=(12, 4))

    score_chart.plot(steps, scores, alpha=0.3, label="Episode reward")
    score_chart.plot(steps[WINDOW - 1:], mean_of_last_games(scores), label=f"Mean of last {WINDOW}")
    score_chart.axhline(random_score, color="red", linestyle="--", label=f"Random agent ({random_score:,.0f})")
    score_chart.set(title="Reward", xlabel="Training steps", ylabel="Reward (game score)")
    score_chart.legend()

    length_chart.plot(steps, lengths, alpha=0.3, label="Episode length")
    length_chart.plot(steps[WINDOW - 1:], mean_of_last_games(lengths), label=f"Mean of last {WINDOW}")
    length_chart.set(title="Episode length", xlabel="Training steps", ylabel="Episode length (steps)")
    length_chart.legend()

    figure.suptitle(Path(run_dir).name)
    figure.tight_layout()
    figure.savefig(output_path, dpi=100)
    plt.close(figure)


def plot_comparison(run_dirs, output_path, random_score):
    """One chart with the mean score of the last WINDOW games for every run, to compare them."""
    figure, chart = plt.subplots(figsize=(10, 5))

    for run_dir in run_dirs:
        steps, scores, _ = read_log(run_dir)
        chart.plot(steps[WINDOW - 1:], mean_of_last_games(scores), label=Path(run_dir).name)

    chart.axhline(random_score, color="red", linestyle="--", label=f"Random agent ({random_score:,.0f})")
    chart.set(title=f"Mean score of the last {WINDOW} games", xlabel="Training steps",
              ylabel="Reward (game score)")
    chart.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=100)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Make training plots from training logs.")
    parser.add_argument("runs", nargs="+", help="one or more run folders that contain training_log.csv")
    parser.add_argument("--output", required=True, help="where to save the image, for example plot.png")
    parser.add_argument("--random-score", type=float, default=17_358,
                        help="score of the random agent (mean of the notebook's 50 seeded games), "
                             "drawn as a red dashed line")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if len(args.runs) == 1:
        plot_one_run(args.runs[0], args.output, args.random_score)
    else:
        plot_comparison(args.runs, args.output, args.random_score)
    print("Saved", args.output)


if __name__ == "__main__":
    main()
