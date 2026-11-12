# Bianca Haguma: Epsilon-schedule experiments

How every run was set up

200,000 steps per run, seed 42

Learning rate 0.0001, gamma 0.99 and batch size 32 are the same in every run

Only the epsilon schedule changes: epsilon start, epsilon end and epsilon decay

Score is the mean real game score of the last 20 games of training

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Score | Observed behaviour |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 1e-4 | 0.99 | 32 | 1.0 | 0.10 | 25,000 steps | 18,815 | Exploration reached epsilon 0.10 early. The score improved late in training, finishing with a final last-20 mean of 18,815, close to the random-play baseline. |
| 2 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 50,000 steps | 18,315 | Score started near 22,000 and fell to a 20-episode mean of about 10,400 near step 78,000, soon after epsilon reached 0.05. It recovered from about 100,000 steps and finished at 18,315, just below random play (19,540). |
| 3 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 150,000 steps | 14,995 | Slow decay kept exploration active through most of training. The final last-20 mean was 14,995. |
| 4 | 1e-4 | 0.99 | 32 | 0.7 | 0.10 | 200,000 steps | 14,295 | Gradual decay kept exploration active for most of training. The score peaked near 20k during the run, then declined as epsilon approached 0.10; final last-20 mean was 14,295. |
| 5 | 1e-4 | 0.99 | 32 | 0.7 | 0.05 | 25,000 steps | 18,870 | Fast decay reached epsilon 0.05 early, followed by a strong late improvement. The final last-20 mean was 18,870, close to the random-play baseline. |
| 6 | 1e-4 | 0.99 | 32 | 0.7 | 0.01 | 50,000 steps | 16,160 | Epsilon dropped quickly to 0.01, then the run improved mid-training before becoming unstable near the end. Final last-20 mean was 16,160. |
| 7 | 1e-4 | 0.99 | 32 | 0.5 | 0.10 | 150,000 steps | 14,650 | Lower starting epsilon and slow decay gave steady exploration, with a late dip before partial recovery. Final last-20 mean was 14,650. |
| 8 | 1e-4 | 0.99 | 32 | 0.5 | 0.05 | 200,000 steps | 13,930 | Slow decay kept exploration active throughout training. The final last-20 mean was 13,930 as epsilon approached 0.05. |
| 9 | 1e-4 | 0.99 | 32 | 0.5 | 0.01 | 25,000 steps | 19,475 | Very fast decay reached epsilon 0.01 early, leaving a long exploitation phase. The score improved late and finished at 19,475, slightly above the random-play baseline. |
| 10 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 50,000 steps | 23,580 | Maximum initial exploration fell to 0.01 quickly, followed by strong late improvement and longer episodes. Final last-20 mean was 23,580, above the random-play baseline. |

