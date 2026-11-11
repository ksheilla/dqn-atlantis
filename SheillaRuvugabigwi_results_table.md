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
