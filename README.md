# End-of-Turn Detection Assignment

This repository contains the materials for a speech turn-taking assignment focused on **end-of-turn (EOT) detection** for voice AI agents.

## Problem Summary

Voice assistants need to decide, at every pause in a user turn, whether the user is:

- still speaking and only pausing briefly, or
- actually done talking.

If the system responds too early, it interrupts the user. If it responds too late, the conversation feels slow and unnatural.

The goal of this assignment is to build a model from scratch that predicts the probability that a pause is the true end of a turn.

## Data

The provided archive `eot_handout.zip` contains the dataset in this structure:

- `eot_data/english/audio/*.wav`
- `eot_data/english/labels.csv`
- `eot_data/hindi/audio/*.wav`
- `eot_data/hindi/labels.csv`

Each WAV file contains one real user turn from a human-to-voice-agent phone conversation. Each silence pause of at least 100 ms inside a turn is annotated in the corresponding `labels.csv` file.

### Label Columns

- `turn_id`: turn identifier matching the WAV filename
- `audio_file`: relative path to the WAV file
- `pause_index`: pause number within the turn
- `pause_start`: time in seconds when speech stops
- `pause_end`: time in seconds when speech resumes, or when the file ends
- `label`: `hold` if the user continues after the pause, `eot` if the pause is the true end of the turn

## Task

For each pause, the model must output `p_eot`, the probability that the turn is over.

### Critical Constraint

The feature set must be causal:

- features for a pause may use only audio from the start of the turn up to `pause_start`
- no audio after the pause may be used

This is a hard requirement because the model is meant to simulate a live agent.

## Deliverables

The assignment requires all of the following:

1. `predict.py` - a script that runs as `python predict.py --data_dir <folder> --out predictions.csv`
2. `predictions.csv` for both the English and Hindi folders
3. `SUMMARY.html` - a detailed HTML report describing the solution, results, graphs, and what was done by the human vs. the coding agent
4. `RUNLOG.md` - notes after every scoring run, including the score and a short description of what changed
5. `NOTES.md` - a short summary of the model signal, failure cases, and next steps

## Scoring

Scoring is based on a live-agent simulation using the model predictions. The reported metric is:

- **Mean response delay (ms) at a false-cutoff rate ≤ 5%**

In practical terms, the best model delays the response as little as possible after the user has actually finished, while keeping false interruptions under control.

The final grade is based on a hidden test set, mostly Hindi, plus the run log and a short discussion of the model.

## Constraints

- CPU only
- no GPUs or cloud training
- allowed libraries: `numpy`, `scipy`, `scikit-learn`, `pandas`, `librosa`, `PyTorch`
- no pretrained models, downloaded weights, or external datasets
- no Whisper, wav2vec, Silero, WebRTC VAD, Hugging Face models, or TTS/ASR APIs

## Suggested Baseline Path

The handout suggests the following workflow:

1. Run the starter silence baseline and score it.
2. Inspect a few examples where silence-only fails.
3. Extract causal prosodic features from the last ~1.5 seconds of speech before each pause.
4. Train a small classifier and score it.
5. Review the worst errors and iterate.
6. Finalize predictions and documentation.

## Goal

The central challenge is to use only information available before the pause and still infer whether the speaker is likely finished. Useful signals may include pause length, pitch, energy, speaking rate, and short-term prosody before the silence.

## Project Scripts

- `python baseline.py --data_dir eot_data/english --out predictions.csv` runs the silence-only baseline.
- `python train.py --data_dir eot_data/english --out predictions.csv` trains the submission model, saves `eot_model.pkl`, and writes predictions.
- `python predict.py --data_dir <folder> --out predictions.csv` loads `eot_model.pkl` and writes predictions for any folder with the assignment schema.
- `python score.py --data_dir <folder> --pred predictions.csv` evaluates predictions with the official metric.

The saved model artifact `eot_model.pkl` is part of the submission package and must be present when running `predict.py`.

