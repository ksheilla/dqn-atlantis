# Hyperparameter experiments: Sheilla Keza RUVUGABIGWI

**Member name:** Sheilla Keza RUVUGABIGWI
**My part:** environment, CNN and replay buffer (`dqn_components.py`), and experiments on learning rate, then batch size at my best learning rate.

## How every run was set up

Only the settings in the table change. Everything else stays the same:

- 200,000 steps per run, seed 42
- Learning after every step from a replay buffer of 100,000 experiences
- Learning starts after 10,000 warm-up steps
- Target network updated every 1,000 steps
- Rewards clipped to -1, 0 or +1 for learning, but the real game score is reported
- Epsilon starts at 1.0, ends at 0.05, and decays linearly over the first 100,000 steps
- Gamma is 0.99 in every run

**Score** is the mean real game score of the last 20 episodes of training. For reference, a random agent averaged 19,540 over 10 episodes. My baseline run (experiment 1) scored 13,205.

Each run's model and training log are in `models/SheillaRuvugabigwi/`, in the folder for that experiment.

## Results

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Score | Observed behaviour |
|---|---|---|---|---|---|---|---|---|
| 1 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 100,000 steps | 13,205 | Baseline. Score fell from about 16k to about 11k once epsilon reached 0.05, then recovered slightly. Below random play. |
| 2 | 1e-3 | 0.99 | 32 | 1.0 | 0.05 | 100,000 steps | 13,530 | Fell to about 9k mid-run, then recovered slowly. No clear learning. |
| 3 | 5e-4 | 0.99 | 32 | 1.0 | 0.05 | 100,000 steps | 17,500 | Dipped to about 13k, then climbed with longer episodes toward the end. Best of the learning-rate sweep. |
| 4 | 2.5e-4 | 0.99 | 32 | 1.0 | 0.05 | 100,000 steps | 12,030 | Slid to about 12k after epsilon reached 0.05 and stayed there, with many short episodes. |
| 5 | 5e-5 | 0.99 | 32 | 1.0 | 0.05 | 100,000 steps | 11,610 | Fell to about 8k, then crept up slightly. Learning rate too small for 200,000 steps. |
| 6 | 1e-5 | 0.99 | 32 | 1.0 | 0.05 | 100,000 steps | 7,160 | Steady decline with very short episodes. Worst run. |
| 7 | 5e-4 | 0.99 | 16 | 1.0 | 0.05 | 100,000 steps | 10,625 | Noisy updates. Fell to about 8k and recovered only slowly. |
| 8 | 5e-4 | 0.99 | 64 | 1.0 | 0.05 | 100,000 steps | 12,930 | Peaked at a last-20 mean of 21,305 around step 83,000, above random play, then declined. Unstable. |
| 9 | 5e-4 | 0.99 | 128 | 1.0 | 0.05 | 100,000 steps | **18,695** | **My best run.** Stayed above random play for about 100,000 steps (peak last-20 mean 26,160 near step 154,000) with longer episodes, then slipped at the end. |
| 10 | 5e-4 | 0.99 | 256 | 1.0 | 0.05 | 100,000 steps | 12,595 | Peaked around 19.5k, then fell to about 12k. About 50 minutes per run, with no gain over batch 128. |

## What I found

**Best: experiment 9** (learning rate 5e-4, batch 128), score 18,695.

- It was the only run that stayed above random play (19,540) for a long stretch after epsilon had dropped, with a last-20 mean of about 20,000 to 24,000 from step 75,000 to 175,000.
- Its games were longer (often 1,500 to 2,400 steps), so the agent survived longer.
- A bigger batch probably gave steadier updates. It still ended just under random play, so it improved but did not learn to play well.

**Worst: experiment 6** (learning rate 1e-5), score 7,160.

- Its score fell as epsilon dropped and never recovered, ending far below random play.
- Its games were the shortest (often 500 to 900 steps).
- The learning rate is probably too small to change the network in 200,000 steps, so the agent plays a poor, almost fixed policy. This is the likely cause, not something I tested.

**Other findings:** learning rate 5e-4 did best in the sweep. Batch 16 was too noisy, and batch 256 gave no gain over 128 for about 2.5 times the run time. Each saved model holds the weights at the end of its run, not at its best moment.

**Limits:** each setting was run once with one seed, so small differences between runs may be luck. All my runs used gamma 0.99, and none ended clearly above random play, so gamma is probably more important than the learning rate or batch size.
