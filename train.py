"""Dev helper that trains the model, saves an artifact, and writes predictions."""

import argparse
import csv
import os

from modeling import (
    fit_model_from_directories,
    held_out_turn_accuracy,
    load_examples,
    predict_with_model,
    save_model,
)


MODEL_PATH = os.path.join(os.path.dirname(__file__), "eot_model.pkl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    X, y, groups, _ = load_examples(args.data_dir)
    print(f"held-out turn accuracy: {held_out_turn_accuracy(X, y, groups):.3f}")

    data_dirs = [
        os.path.join(os.path.dirname(__file__), "eot_data", "english"),
        os.path.join(os.path.dirname(__file__), "eot_data", "hindi"),
    ]
    model = fit_model_from_directories(data_dirs)
    save_model(model, MODEL_PATH)

    probs, keys = predict_with_model(model, args.data_dir)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["turn_id", "pause_index", "p_eot"])
        for (turn_id, pause_index), prob in zip(keys, probs):
            writer.writerow([turn_id, pause_index, f"{float(prob):.6f}"])
    print(f"wrote {len(keys)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
