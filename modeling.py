"""Shared feature extraction and model training for EOT detection."""

import csv
import pickle
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import f0_contour, frame_energy_db, load_wav, speech_before


def _safe_mean(values):
    return float(np.mean(values)) if len(values) else 0.0


def _safe_std(values):
    return float(np.std(values)) if len(values) else 0.0


def _safe_min(values):
    return float(np.min(values)) if len(values) else 0.0


def _safe_max(values):
    return float(np.max(values)) if len(values) else 0.0


def _safe_quantile(values, q):
    return float(np.quantile(values, q)) if len(values) else 0.0


def _linear_slope(values):
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=np.float32)
    x = x - x.mean()
    y = np.asarray(values, dtype=np.float32)
    y = y - y.mean()
    denom = float(np.sum(x * x))
    if denom <= 0:
        return 0.0
    return float(np.sum(x * y) / denom)


def _tail_mean(values, n):
    if len(values) == 0:
        return 0.0
    return float(np.mean(values[-min(n, len(values)) :]))


def _head_mean(values, n):
    if len(values) == 0:
        return 0.0
    return float(np.mean(values[: min(n, len(values))]))


def _trailing_voiced_run(f0):
    count = 0
    for value in f0[::-1]:
        if value > 0:
            count += 1
        else:
            break
    return float(count)


def _leading_voiced_run(f0):
    count = 0
    for value in f0:
        if value > 0:
            count += 1
        else:
            break
    return float(count)


class EOTEnsembleModel:
    def __init__(self):
        self.logistic = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", C=0.6),
        )
        self.forest = RandomForestClassifier(
            n_estimators=350,
            max_depth=7,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=0,
            n_jobs=-1,
        )

    def fit(self, X, y):
        self.logistic.fit(X, y)
        self.forest.fit(X, y)
        return self

    def predict_proba(self, X):
        p1 = self.logistic.predict_proba(X)[:, 1]
        p2 = self.forest.predict_proba(X)[:, 1]
        return np.column_stack([1.0 - (p1 + p2) / 2.0, (p1 + p2) / 2.0])

    def score(self, X, y):
        return float(np.mean((self.predict_proba(X)[:, 1] >= 0.5) == y))


def extract_features(x, sr, pause_start, window_s=1.5):
    """Causal features from audio strictly before pause_start."""

    seg = speech_before(x, sr, pause_start, window_s=window_s)
    if len(seg) == 0:
        return np.zeros(39, dtype=np.float32)

    e = frame_energy_db(seg, sr)
    f0 = f0_contour(seg, sr)
    voiced = f0[f0 > 0]
    n_frames = len(e)
    voiced_tail = voiced[-min(10, len(voiced)) :] if len(voiced) else np.array([], dtype=np.float32)
    voiced_head = voiced[: min(10, len(voiced))] if len(voiced) else np.array([], dtype=np.float32)

    energy_first = _head_mean(e, 5)
    energy_last = _tail_mean(e, 5)
    energy_prev = _head_mean(e[:-5], min(5, max(0, len(e) - 5))) if len(e) > 5 else energy_first
    energy_tail_10 = _tail_mean(e, 10)
    energy_slope = _linear_slope(e)
    energy_tail_slope = _linear_slope(e[-10:]) if len(e) >= 2 else 0.0

    voiced_ratio = float(len(voiced) / len(f0)) if len(f0) else 0.0
    f0_first = float(voiced[0]) if len(voiced) else 0.0
    f0_last = float(voiced[-1]) if len(voiced) else 0.0
    f0_last5 = _tail_mean(voiced, 5)
    f0_mean = _safe_mean(voiced)
    f0_std = _safe_std(voiced)
    f0_min = _safe_min(voiced)
    f0_max = _safe_max(voiced)
    f0_slope = _linear_slope(voiced)
    f0_q25 = _safe_quantile(voiced, 0.25)
    f0_q50 = _safe_quantile(voiced, 0.50)
    f0_q75 = _safe_quantile(voiced, 0.75)
    e_q10 = _safe_quantile(e, 0.10)
    e_q25 = _safe_quantile(e, 0.25)
    e_q50 = _safe_quantile(e, 0.50)
    e_q75 = _safe_quantile(e, 0.75)
    e_q90 = _safe_quantile(e, 0.90)

    features = np.array(
        [
            pause_start,
            np.log1p(pause_start),
            len(seg) / sr,
            len(seg),
            len(seg) / max(1.0, pause_start * sr),
            len(e),
            _safe_mean(e),
            _safe_std(e),
            _safe_min(e),
            _safe_max(e),
            e_q10,
            e_q25,
            e_q50,
            e_q75,
            e_q90,
            energy_first,
            energy_last,
            energy_last - energy_first,
            energy_prev,
            energy_tail_10,
            _tail_mean(e, 3),
            _tail_mean(e, 7),
            energy_slope,
            energy_tail_slope,
            len(f0),
            voiced_ratio,
            len(voiced),
            _leading_voiced_run(f0),
            f0_mean,
            f0_std,
            f0_min,
            f0_max,
            f0_q25,
            f0_q50,
            f0_q75,
            f0_first,
            f0_last,
            f0_last5,
            f0_slope,
            _safe_mean(voiced_head),
            _safe_mean(voiced_tail),
            _trailing_voiced_run(f0),
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def load_examples(data_dir):
    labels_path = os.path.join(data_dir, "labels.csv")
    if not os.path.exists(labels_path):
        raise SystemExit(f"missing labels.csv in {data_dir}")

    rows = list(csv.DictReader(open(labels_path, newline="")))
    cache = {}
    X, y, groups, keys = [], [], [], []
    for row in rows:
        path = os.path.join(data_dir, row["audio_file"])
        if path not in cache:
            cache[path] = load_wav(path)
        x, sr = cache[path]
        X.append(extract_features(x, sr, float(row["pause_start"])))
        y.append(1 if row["label"] == "eot" else 0)
        groups.append(row["turn_id"])
        keys.append((row["turn_id"], row["pause_index"]))
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64), groups, keys


def make_model():
    return EOTEnsembleModel()


def fit_model_from_directories(data_dirs):
    X_parts, y_parts, group_parts = [], [], []
    for data_dir in data_dirs:
        X, y, groups, _ = load_examples(data_dir)
        X_parts.append(X)
        y_parts.append(y)
        group_parts.extend([(data_dir, group) for group in groups])

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    model = make_model()
    model.fit(X, y)
    return model


def save_model(model, model_path):
    with open(model_path, "wb") as f:
        pickle.dump(model, f)


def load_model(model_path):
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict_with_model(model, data_dir):
    X, _, _, keys = load_examples(data_dir)
    probs = model.predict_proba(X)[:, 1]
    return probs, keys


def fit_and_predict(data_dir):
    X, y, groups, keys = load_examples(data_dir)
    model = make_model()
    model.fit(X, y)
    probs = model.predict_proba(X)[:, 1]
    return probs, keys, model, (X, y, groups)


def held_out_turn_accuracy(X, y, groups):
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    model = make_model()
    model.fit(X[train_idx], y[train_idx])
    return float(model.score(X[test_idx], y[test_idx]))
