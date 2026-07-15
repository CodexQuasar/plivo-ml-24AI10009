# RUNLOG

## 2026-07-15

- English split: AUC 0.891, best mean response delay 742 ms at 5.0% interrupted turns.
- Hindi split: AUC 0.941, best mean response delay 586 ms at 5.0% interrupted turns.
- Added richer causal features: pause timing, pause-position context, distributional energy/F0 stats, and voiced-run structure.
- Switched to a small ensemble classifier and retrained the submission model on the provided labeled data.
