"""Audio utilities for the EOT assignment.

These helpers only expose causal audio access and simple signal processing.
Feature design belongs in the modeling code.
"""

import numpy as np
from scipy.io import wavfile

FRAME_MS = 25
HOP_MS = 10


def load_wav(path):
    sr, x = wavfile.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        scale = max(abs(np.iinfo(x.dtype).min), np.iinfo(x.dtype).max)
        x = x.astype(np.float32) / float(scale)
    else:
        x = x.astype(np.float32)
    return x, int(sr)


def speech_before(x, sr, pause_start, window_s=1.5):
    end = int(pause_start * sr)
    start = max(0, end - int(window_s * sr))
    return x[start:end]


def frames(x, sr, frame_ms=FRAME_MS, hop_ms=HOP_MS):
    fl = int(sr * frame_ms / 1000)
    hp = int(sr * hop_ms / 1000)
    if len(x) < fl:
        return np.empty((0, fl), dtype=np.float32)
    n = 1 + (len(x) - fl) // hp
    idx = np.arange(fl)[None, :] + hp * np.arange(n)[:, None]
    return x[idx]


def frame_energy_db(x, sr):
    fr = frames(x, sr)
    if len(fr) == 0:
        return np.empty(0, dtype=np.float32)
    rms = np.sqrt(np.mean(fr**2, axis=1) + 1e-12)
    return 20.0 * np.log10(rms + 1e-12)


def autocorr_f0(frame, sr, fmin=60.0, fmax=400.0, voicing_thresh=0.30):
    frame = frame - np.mean(frame)
    if np.max(np.abs(frame)) < 1e-4:
        return 0.0
    ac = np.correlate(frame, frame, mode="full")[len(frame) - 1 :]
    if ac[0] <= 0:
        return 0.0
    ac = ac / ac[0]
    lo = int(sr / fmax)
    hi = min(int(sr / fmin), len(ac) - 1)
    if hi <= lo:
        return 0.0
    lag = lo + int(np.argmax(ac[lo:hi]))
    if ac[lag] < voicing_thresh:
        return 0.0
    return float(sr / lag)


def f0_contour(x, sr, frame_ms=40, hop_ms=HOP_MS):
    fr = frames(x, sr, frame_ms=frame_ms, hop_ms=hop_ms)
    if len(fr) == 0:
        return np.empty(0, dtype=np.float32)
    return np.array([autocorr_f0(f, sr) for f in fr], dtype=np.float32)
