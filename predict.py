"""Assignment entrypoint.

Usage:
    python predict.py --data_dir <folder> --out predictions.csv
"""

import argparse
import csv
import os

from modeling import load_model, predict_with_model


MODEL_PATH = os.path.join(os.path.dirname(__file__), "eot_model.pkl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    if not os.path.exists(MODEL_PATH):
        raise SystemExit(
            f"missing trained model artifact: {MODEL_PATH}. Run train.py first to create it."
        )

    model = load_model(MODEL_PATH)
    probs, keys = predict_with_model(model, args.data_dir)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["turn_id", "pause_index", "p_eot"])
        for (turn_id, pause_index), prob in zip(keys, probs):
            writer.writerow([turn_id, pause_index, f"{float(prob):.6f}"])
    print(f"wrote {len(keys)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
