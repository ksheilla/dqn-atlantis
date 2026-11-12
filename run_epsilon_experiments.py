"""Train ten agents, each with a different epsilon schedule, and collect their final scores in one table."""

import csv
import json
import subprocess
import sys
from pathlib import Path


EXPERIMENTS = [
    {"epsilon_start": 1.0, "epsilon_end": 0.10, "epsilon_decay": 25_000},
    {"epsilon_start": 1.0, "epsilon_end": 0.05, "epsilon_decay": 50_000},
    {"epsilon_start": 1.0, "epsilon_end": 0.01, "epsilon_decay": 150_000},
    {"epsilon_start": 0.7, "epsilon_end": 0.10, "epsilon_decay": 200_000},
    {"epsilon_start": 0.7, "epsilon_end": 0.05, "epsilon_decay": 25_000},
    {"epsilon_start": 0.7, "epsilon_end": 0.01, "epsilon_decay": 50_000},
    {"epsilon_start": 0.5, "epsilon_end": 0.10, "epsilon_decay": 150_000},
    {"epsilon_start": 0.5, "epsilon_end": 0.05, "epsilon_decay": 200_000},
    {"epsilon_start": 0.5, "epsilon_end": 0.01, "epsilon_decay": 25_000},
    {"epsilon_start": 1.0, "epsilon_end": 0.01, "epsilon_decay": 50_000},
]


def validate_experiments():
    baseline = (1.0, 0.05, 100_000)
    schedules = set()

    for experiment in EXPERIMENTS:
        schedule = (
            experiment["epsilon_start"],
            experiment["epsilon_end"],
            experiment["epsilon_decay"],
        )
        if schedule == baseline:
            raise ValueError("An experiment matches the baseline epsilon schedule.")
        if schedule in schedules:
            raise ValueError("Every experiment must use a different epsilon schedule.")
        schedules.add(schedule)


def run_experiment(experiment_number, settings, output_root, common_args):
    output_dir = output_root / f"exp{experiment_number:02d}"
    command = [
        sys.executable,
        "train.py",
        *common_args,
        "--epsilon-start", str(settings["epsilon_start"]),
        "--epsilon-end", str(settings["epsilon_end"]),
        "--epsilon-decay", str(settings["epsilon_decay"]),
        "--output-dir", str(output_dir),
    ]
    subprocess.run(command, check=True, cwd=Path(__file__).parent)

    with open(output_dir / "summary.json", encoding="utf-8") as summary_file:
        summary = json.load(summary_file)
    return summary


def write_results(results, output_root):
    fieldnames = [
        "experiment",
        "epsilon_start",
        "epsilon_end",
        "epsilon_decay",
        "last_20_mean_reward",
        "observed_behaviour",
    ]
    with open(output_root / "experiment_results.csv", "w", newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    with open(output_root / "experiment_results.md", "w", encoding="utf-8") as result_file:
        result_file.write("# Epsilon schedule experiments\n\n")
        result_file.write("| Experiment | Epsilon start | Epsilon end | Epsilon decay | ")
        result_file.write("Last-20 mean reward | Observed behaviour |\n")
        result_file.write("|---:|---:|---:|---:|---:|---|\n")
        for result in results:
            score = result["last_20_mean_reward"]
            score_text = "N/A" if score is None else f"{score:.0f}"
            # The observed behaviour column is left empty to be filled in by hand.
            result_file.write(
                f"| {result['experiment']} | {result['epsilon_start']} | "
                f"{result['epsilon_end']} | {result['epsilon_decay']:,} | "
                f"{score_text} |  |\n"
            )


def main():
    validate_experiments()
    output_root = Path(__file__).parent / "models" / "epsilon_experiments"
    output_root.mkdir(parents=True, exist_ok=True)

    common_args = [
        "--total-steps", "200000",
        "--seed", "42",
        "--lr", "0.0001",
        "--gamma", "0.99",
        "--batch-size", "32",
    ]
    results = []
    for experiment_number, settings in enumerate(EXPERIMENTS, start=1):
        summary = run_experiment(experiment_number, settings, output_root, common_args)
        results.append({
            "experiment": experiment_number,
            **settings,
            "last_20_mean_reward": summary["last_20_mean_reward"],
            "observed_behaviour": "",
        })
        write_results(results, output_root)

    print(f"Completed {len(results)} experiments. Results are in {output_root}.")


if __name__ == "__main__":
    main()
