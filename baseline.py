"""Silence-only baseline: every pause looks like end-of-turn (p_eot = 1)."""

import argparse
import csv
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    rows = []
    with open(os.path.join(args.data_dir, "labels.csv"), newline="") as f:
        for row in csv.DictReader(f):
            rows.append({"turn_id": row["turn_id"], "pause_index": row["pause_index"], "p_eot": 1.0})
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["turn_id", "pause_index", "p_eot"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
