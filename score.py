"""Official scorer: simulates a live voice agent using p_eot scores."""

import argparse
import csv
import os

import numpy as np

TIMEOUT_S = 1.6
THRESHOLDS = np.round(np.arange(0.05, 1.0, 0.05), 3)
DELAYS = np.round(np.arange(0.10, 1.65, 0.05), 3)


def load(labels_csv, pred_csv):
    preds = {}
    with open(pred_csv, newline="") as f:
        for row in csv.DictReader(f):
            preds[(row["turn_id"], int(row["pause_index"]))] = float(row["p_eot"])
    pauses = []
    with open(labels_csv, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["turn_id"], int(row["pause_index"]))
            if key not in preds:
                raise SystemExit(f"missing prediction for {key}")
            pauses.append(
                {
                    "turn_id": row["turn_id"],
                    "dur": float(row["pause_end"]) - float(row["pause_start"]),
                    "label": row["label"],
                    "p": preds[key],
                }
            )
    return pauses


def evaluate(pauses, threshold, delay):
    turns_cut = set()
    turn_ids = set()
    latencies = []
    for pause in pauses:
        turn_ids.add(pause["turn_id"])
        fires = pause["p"] >= threshold
        if pause["label"] == "hold":
            if fires and delay < pause["dur"]:
                turns_cut.add(pause["turn_id"])
        else:
            latencies.append(delay if fires else TIMEOUT_S)
    cutoff_rate = len(turns_cut) / max(1, len(turn_ids))
    return cutoff_rate, float(np.mean(latencies)) if latencies else TIMEOUT_S


def score(labels_csv, pred_csv, budget=0.05):
    pauses = load(labels_csv, pred_csv)
    best = None
    for threshold in THRESHOLDS:
        for delay in DELAYS:
            cutoff, latency = evaluate(pauses, threshold, delay)
            if cutoff <= budget and (best is None or latency < best["latency"]):
                best = {"latency": latency, "cutoff": cutoff, "threshold": threshold, "delay": delay}
    if best is None:
        best = {"latency": TIMEOUT_S, "cutoff": 0.0, "threshold": 1.0, "delay": TIMEOUT_S}

    y = np.array([1 if pause["label"] == "eot" else 0 for pause in pauses])
    s = np.array([pause["p"] for pause in pauses])
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), len(y) - y.sum()
    auc = ((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")
    best["auc"] = float(auc)
    best["n_turns"] = len({pause["turn_id"] for pause in pauses})
    best["n_pauses"] = len(pauses)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--budget", type=float, default=0.05)
    args = ap.parse_args()
    result = score(os.path.join(args.data_dir, "labels.csv"), args.pred, args.budget)
    print(f"turns={result['n_turns']}  pauses={result['n_pauses']}  AUC={result['auc']:.3f}")
    print(f"BEST @ <= {int(args.budget * 100)}% interrupted turns:")
    print(f"  mean response delay : {result['latency'] * 1000:.0f} ms   <-- your score, lower is better")
    print(f"  interrupted turns   : {result['cutoff'] * 100:.1f}%")
    print(f"  operating point     : threshold={result['threshold']}, delay={result['delay'] * 1000:.0f} ms")


if __name__ == "__main__":
    main()
