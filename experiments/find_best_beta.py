# Copyright (c) 2023, Yamagishi Laboratory, National Institute of Informatics
# Author: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.
#
# This code contains many hard-coded paths. It assumes that
# eval.dev.txt files are in {dataname}_{loss}/roberta-base-128-{beta}-out.


import sys
from pathlib import Path


def get_acc_ece(fname):
    filepath = Path(fname)
    assert filepath.exists(), filepath
    lines = [line.strip() for line in open(filepath, "r")]
    acc, ece = lines[-2].lower(), lines[-1].lower()
    assert "acc: " == acc[:5] and "ece: " == ece[:5]
    return float(acc[5:]), float(ece[5:])


def main(dataname, loss, pretrained="roberta-base", split="dev"):
    abs_path = Path(__file__).parent.resolve()
    base_loss = "cross_entropy"
    filepath = f"{abs_path}/{dataname}_{base_loss}/{pretrained}-128-out/eval.{split}.txt"  # noqa: E501
    acc, ece = get_acc_ece(filepath)
    base_acc, base_ece = acc, ece
    print(f"{base_loss}: ACC={base_acc}, ECE={base_ece}")

    results = []
    betas = [float(line.strip()) for line in open(f"{abs_path}/beta.txt", "r")]
    for beta in betas:
        filepath = f"{abs_path}/{dataname}_{loss}/{pretrained}-128-{beta:.2f}-out/eval.{split}.txt"  # noqa: E501
        acc, ece = get_acc_ece(filepath)
        results.append((beta, acc, ece))

    print(f"{loss}: {results}")

    # Keep results having lower/equal ece than base
    results_to_keep = [
        (beta, acc, ece) for (beta, acc, ece) in results if ece <= base_ece
    ]

    # If empty, keep ones having ece within margin 5
    if not len(results_to_keep):
        results_to_keep = [
            (beta, acc, ece) for (beta, acc, ece) in results if abs(ece - base_ece) <= 5
        ]

    assert len(results_to_keep)
    print(f"filtered: {results_to_keep}")

    # Sort by acc
    sorted_results = sorted(results_to_keep, key=lambda x: x[1], reverse=True)
    print(f"sorted: {sorted_results}")

    best_beta = None

    # Find the one with better acc, lower/equal ece
    for (beta, acc, ece) in sorted_results:
        if base_acc < acc and base_ece >= ece:
            best_beta = beta
            break

    # If not found, just better/equal acc, lower/equal ece
    if best_beta is None:
        for (beta, acc, ece) in sorted_results:
            if base_acc <= acc and base_ece >= ece:
                best_beta = beta
                break

    # If not found, just better/equal acc
    if best_beta is None:
        for (beta, acc, ece) in sorted_results:
            if base_acc <= acc:  # just better/equal acc
                best_beta = beta
                break

    # If not found, just best acc
    if best_beta is None:
        best_beta = sorted_results[0][0]

    print(f"best_beta: {best_beta}")

    filepath = f"{abs_path}/{dataname}_{loss}/best_beta.txt"
    with open(filepath, "w") as f:
        f.write(f"{best_beta:.2f}\n")
    print(f"Save to {filepath}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
