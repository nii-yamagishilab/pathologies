# Modified from https://github.com/torrvision/focal_calibration/blob/main/Metrics/metrics.py

import argparse
import jsonlines
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from processors import FactVerificationProcessor
from matplotlib.offsetbox import AnchoredText
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

COUNT = "count"
CONF = "conf"
ACC = "acc"
BIN_ACC = "bin_acc"
BIN_CONF = "bin_conf"


def bin_initializer(bin_dict, num_bins=10):
    for i in range(num_bins):
        bin_dict[i][COUNT] = 0
        bin_dict[i][CONF] = 0
        bin_dict[i][ACC] = 0
        bin_dict[i][BIN_ACC] = 0
        bin_dict[i][BIN_CONF] = 0


def populate_bins(confs, preds, labels, num_bins=10):
    bin_dict = {}
    for i in range(num_bins):
        bin_dict[i] = {}
    bin_initializer(bin_dict, num_bins)
    num_test_samples = len(confs)

    for i in range(0, num_test_samples):
        confidence = confs[i]
        prediction = preds[i]
        label = labels[i]
        binn = int(math.ceil(((num_bins * confidence) - 1)))
        bin_dict[binn][COUNT] = bin_dict[binn][COUNT] + 1
        bin_dict[binn][CONF] = bin_dict[binn][CONF] + confidence
        bin_dict[binn][ACC] = bin_dict[binn][ACC] + (1 if (label == prediction) else 0)

    for binn in range(0, num_bins):
        if bin_dict[binn][COUNT] == 0:
            bin_dict[binn][BIN_ACC] = 0
            bin_dict[binn][BIN_CONF] = 0
        else:
            bin_dict[binn][BIN_ACC] = float(bin_dict[binn][ACC]) / bin_dict[binn][COUNT]
            bin_dict[binn][BIN_CONF] = bin_dict[binn][CONF] / float(
                bin_dict[binn][COUNT]
            )
    return bin_dict


def plot_confidence_histogram(confs, preds, labels, out_file, num_bins=10):
    bin_dict = populate_bins(confs, preds, labels, num_bins)
    bns = [(i / float(num_bins)) for i in range(num_bins)]
    num_samples = len(labels)
    y = []
    for i in range(num_bins):
        n = (bin_dict[i][COUNT] / float(num_samples)) * 100
        y.append(n)

    plt.figure(figsize=(10, 8))
    plt.rcParams.update({"font.size": 20})
    plt.xlim([0.0, 1.0])
    plt.bar(
        bns,
        y,
        align="edge",
        width=0.05,
        color="red",
        alpha=0.5,
        label="% samples",
        edgecolor="black",
    )
    plt.ylabel("% of samples")
    plt.xlabel("Confidence")

    out_file = Path(out_file).with_suffix(".conf.pdf")
    plt.tight_layout()
    plt.savefig(out_file)
    print(f"Saved confidence histogram to {out_file}")


def plot_reliability_diagram(confs, preds, labels, out_file, num_bins=20):
    bin_dict = populate_bins(confs, preds, labels, num_bins)
    bns = [(i / float(num_bins)) for i in range(num_bins)]
    num_samples = len(labels)
    ece = 0
    y = []
    for i in range(num_bins):
        bin_accuracy = bin_dict[i][BIN_ACC]
        bin_confidence = bin_dict[i][BIN_CONF]
        bin_count = bin_dict[i][COUNT]
        ece += (float(bin_count) / num_samples) * abs(bin_accuracy - bin_confidence)
        y.append(bin_accuracy)

    ece = ece * 100.0
    acc = accuracy_score(labels, preds) * 100.0

    plt.figure(figsize=(10, 8))
    plt.rcParams.update({"font.size": 20})
    plt.xlim([0.0, 1.0])
    plt.bar(
        bns,
        bns,
        align="edge",
        width=0.05,
        color="silver",
        label="Expected",
        edgecolor="black",
    )
    plt.bar(
        bns,
        y,
        align="edge",
        width=0.05,
        color="red",
        alpha=0.5,
        label="Actual",
        edgecolor="black",
    )
    plt.ylabel("Accuracy")
    plt.xlabel("Confidence")
    plt.legend()

    box_text = "\n".join([f"ACC = {acc:.1f}", f"ECE = {ece:.1f}"])
    text_box = AnchoredText(box_text, frameon=True, loc=4, pad=0.5)
    plt.setp(text_box.patch, facecolor="white", alpha=0.7)
    plt.gca().add_artist(text_box)

    out_file = Path(out_file).with_suffix(".ece.pdf")
    plt.tight_layout()
    plt.savefig(out_file)
    print(f"Saved reliability diagram to {out_file}")


def expected_calibration_error(confs, preds, labels, num_bins=20):
    bin_dict = populate_bins(confs, preds, labels, num_bins)
    num_samples = len(labels)
    ece = 0
    for i in range(num_bins):
        bin_accuracy = bin_dict[i][BIN_ACC]
        bin_confidence = bin_dict[i][BIN_CONF]
        bin_count = bin_dict[i][COUNT]
        ece += (float(bin_count) / num_samples) * abs(bin_accuracy - bin_confidence)
    return ece * 100.0


def compute_metics(confidence_vals, pred_labels, gold_labels, labels):
    prec = (
        precision_score(
            gold_labels, pred_labels, labels=labels, average=None, zero_division=0
        )
        * 100.0
    )
    rec = (
        recall_score(
            gold_labels, pred_labels, labels=labels, average=None, zero_division=0
        )
        * 100.0
    )
    f1 = (
        f1_score(gold_labels, pred_labels, labels=labels, average=None, zero_division=0)
        * 100.0
    )
    acc = accuracy_score(gold_labels, pred_labels) * 100.0
    mat = confusion_matrix(gold_labels, pred_labels, labels=labels)
    return prec, rec, f1, acc, mat


def compute_avg_len(filepath, key="claim_tokens"):
    len_list = []
    for line in jsonlines.open(filepath):
        n_toks = len(line[key].split())
        len_list.append(n_toks)

    return np.asarray(len_list).mean()


def read_pred_ids(filepath):
    probs = np.loadtxt(filepath, dtype=np.float64)
    ids = np.argmax(probs, axis=1)
    confidence_vals = probs[np.arange(len(probs)), ids]
    return ids, confidence_vals


def read_gold_labels(filepath):
    return [line["label"][0] for line in jsonlines.open(filepath)]


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_file", type=str, required=True)
    parser.add_argument("--prob_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--reduced_file", type=str, default=None)
    parser.add_argument("--num_bins", type=int, default=20)
    parser.add_argument("--plot_diagram", action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = build_args()
    gold_labels = read_gold_labels(args.gold_file)
    pred_ids, confidence_vals = read_pred_ids(args.prob_file)
    labels = FactVerificationProcessor().get_labels(Path(args.gold_file))
    id2label = {i: label for i, label in enumerate(labels)}
    pred_labels = [id2label[i] for i in pred_ids]
    prec, rec, f1, acc, mat = compute_metics(
        confidence_vals, pred_labels, gold_labels, labels
    )

    if args.reduced_file:
        avg_len = compute_avg_len(args.reduced_file)

    ece = expected_calibration_error(
        confidence_vals, pred_labels, gold_labels, num_bins=args.num_bins
    )

    if args.plot_diagram:
        plot_reliability_diagram(
            confidence_vals,
            pred_labels,
            gold_labels,
            args.out_file,
            num_bins=args.num_bins,
        )

        plot_confidence_histogram(
            confidence_vals,
            pred_labels,
            gold_labels,
            args.out_file,
            num_bins=args.num_bins,
        )

    mat = pd.DataFrame(mat, columns=labels, index=labels)
    tab = pd.DataFrame([prec, rec, f1], columns=labels, index=["Prec:", "Rec:", "F1:"])

    results = []
    if args.reduced_file:
        results.append(f"Avg. claim length after reduction: {avg_len.round(1)}")
    results.extend(
        [
            f"{mat}",
            f"{tab.round(1)}",
            f"ACC: {acc.round(1)}",
            f"ECE: {ece.round(1)}",
        ]
    )

    results = "\n".join(results)

    with open(args.out_file, "w") as f:
        f.write(f"{results}\n")
        print(f"Saved results to {args.out_file}")

    print(results)


if __name__ == "__main__":
    main()
