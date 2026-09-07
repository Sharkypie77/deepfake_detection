"""Repeatable held-out evaluation entry point.

The script requires a labeled manifest with `path,label` rows. It reports
measured metrics and writes no provenance manifest automatically.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


def eer(labels, scores):
    false_positive, true_positive, _ = roc_curve(labels, scores)
    index = np.nanargmin(np.abs(false_positive - (1 - true_positive)))
    return float((false_positive[index] + 1 - true_positive[index]) / 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="CSV containing label and score columns")
    args = parser.parse_args()
    with Path(args.predictions).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    labels = np.array([int(row["label"]) for row in rows])
    scores = np.array([float(row["score"]) for row in rows])
    predictions = (scores >= 0.5).astype(int)
    print({
        "samples": len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        "auc": float(roc_auc_score(labels, scores)),
        "eer": eer(labels, scores),
    })


if __name__ == "__main__":
    main()
