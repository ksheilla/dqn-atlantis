# Bianca Haguma: Epsilon-schedule experiments

How every run was set up

200,000 steps per run, seed 42

Learning rate 0.0001, gamma 0.99 and batch size 32 are the same in every run

Only the epsilon schedule changes: epsilon start, epsilon end and epsilon decay

Score is the mean real game score of the last 20 games of training

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Score | Observed behaviour |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 1e-4 | 0.99 | 32 | 1.0 | 0.10 | 25,000 steps | 18,815 | Exploration reached epsilon 0.10 early. The score improved late in training, finishing with a final last-20 mean of 18,815, close to the random-play baseline. |
| 2 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 50,000 steps | 18,315 | Score started near 22,000 and fell to a 20-episode mean of about 10,400 near step 78,000, soon after epsilon reached 0.05. It recovered from about 100,000 steps and finished at 18,315, just above random play (17,358). |
| 3 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 150,000 steps | 11,175 | Score stayed near random play (20-episode mean peaked at 20,750 near step 46,000) while epsilon was still high. It fell to about 12,000 to 15,000 as epsilon dropped and finished at its lowest point, 11,175, below the team baseline. |
| 4 | 1e-4 | 0.99 | 32 | 0.7 | 0.10 | 200,000 steps | 13,605 | Epsilon was still about 0.10 at the end because the decay lasted the whole run. The 20-episode mean peaked at 18,620 near step 48,000, then stayed between about 13,000 and 17,000 and finished at 13,605, just above the team baseline. |
| 5 | 1e-4 | 0.99 | 32 | 0.7 | 0.05 | 25,000 steps | 13,965 | Epsilon reached 0.05 at about step 26,000. The 20-episode mean fell from about 12,000 to a low of 7,635 near step 48,000, then climbed slowly to a peak of 14,740 near step 181,000 and finished at 13,965, slightly above the team baseline. |
| 6 | 1e-4 | 0.99 | 32 | 0.7 | 0.01 | 50,000 steps | 16,160 | Epsilon dropped quickly to 0.01, then the run improved mid-training before becoming unstable near the end. Final last-20 mean was 16,160. |
| 7 | 1e-4 | 0.99 | 32 | 0.5 | 0.10 | 150,000 steps | 14,650 | Lower starting epsilon and slow decay gave steady exploration, with a late dip before partial recovery. Final last-20 mean was 14,650. |
| 8 | 1e-4 | 0.99 | 32 | 0.5 | 0.05 | 200,000 steps | 13,930 | Slow decay kept exploration active throughout training. The final last-20 mean was 13,930 as epsilon approached 0.05. |
| 9 | 1e-4 | 0.99 | 32 | 0.5 | 0.01 | 25,000 steps | 19,475 | Very fast decay reached epsilon 0.01 early, leaving a long exploitation phase. The score improved late and finished at 19,475, a little above the random-play baseline. |
| 10 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 50,000 steps | 23,580 | Maximum initial exploration fell to 0.01 quickly, followed by strong late improvement and longer episodes. Final last-20 mean was 23,580, above the random-play baseline. |


## What I found

**Best:** experiment 10 (epsilon 1.0 → 0.01 over 50,000 steps), score 23,580. It finished furthest above random play (17,358): experiments 9, 1 and 2 also finished above it, but by only 957 to 2,117 points. It had the longest games (mean 1,623 steps over its last 20 episodes). Its score was still rising when training stopped.

**Worst:** experiment 3 (epsilon 1.0 → 0.01 over 150,000 steps), score 11,175, below the team baseline of 13,205.

**Scores in order:** 10 (23,580), 9 (19,475), 1 (18,815), 2 (18,315), 6 (16,160), 7 (14,650), 5 (13,965), 8 (13,930), 4 (13,605), 3 (11,175). The average of all ten was 16,367.

**What helped**

- **Decay length.** All four slow decays (150,000 and 200,000 steps: experiments 3, 4, 7 and 8) scored 14,650 or lower. Five of the six runs with a 25,000 or 50,000-step decay scored 16,160 or higher. The exception was experiment 5 (13,965). The average was 19,352 for 50,000 steps and 17,418 for 25,000 steps, against 13,768 for 200,000 and 12,912 for 150,000. A slow decay keeps epsilon high for most of training, so the agent spends too long acting randomly.
- **A low final epsilon (0.01).** The average was 17,598 for 0.01, 15,690 for 0.10 and 15,403 for 0.05. This gap is small, and the runs with 0.01 include both the best and the worst run, so I would not call it a clear effect.
- **Comparisons where only one setting changes** (all against experiment 10):
  - Final epsilon: experiment 2 (0.05) scored 18,315 against 23,580 for experiment 10 (0.01).
  - Starting epsilon: experiment 6 (start 0.7) scored 16,160 against 23,580 for experiment 10 (start 1.0).
  - Decay length: experiment 3 (150,000 steps) scored 11,175 against 23,580 for experiment 10 (50,000 steps).

**What hurt**

- Slow decay (experiments 3, 4, 7 and 8), as above.
- A starting epsilon of 0.7 had the lowest average (14,577), against 16,018 for 0.5 and 17,971 for 1.0. This is a weak pattern with only three or four runs per group.

**Comparison with the baseline:** nine of the ten schedules scored above the team baseline of 13,205, but three of them (experiments 4, 5 and 8) beat it by less than 800 points. Experiment 3 scored below it.

**Limits**

- Each configuration was run once per session with one seed (42). All ten configurations were run a second time in another session, and the same configuration differed by up to about 5,400 points (experiment 5: 13,965 against 19,375; experiment 9: 19,475 against 14,290; experiment 10: 23,580 against 19,070). Differences of a few thousand points between schedules are therefore within run-to-run variation.
- Two patterns held in both sessions: experiment 10 was among the two highest scores, and runs with a slow decay scored lower on average. The size of the decay gap changed, from about 6,000 points in the stored runs to about 2,400 in the other session.
- Each run changed several epsilon settings at once, so only the three comparisons above isolate one setting. With more time I would change one setting at a time and use several seeds.

**My final choice:** epsilon start 1.0, end 0.01, decay 50,000 steps (experiment 10). It had the highest score and the longest games, and it was still improving at the end.
