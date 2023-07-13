# Copyright (c) 2023, Yamagishi Laboratory, National Institute of Informatics
# Author: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.
#
# This code contains many hard-coded paths. It assumes that
# test.reduced.jsonl files are in covidfact_{loss}/roberta-base-128-{beta}-out.


import jsonlines
import logging
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer
from build_reduced_data import sanity_check

logger = logging.getLogger(__name__)


def clean_tokens(tokens, tokenizer):
    cleaned_toks = []
    for tok in tokens.split():
        cleaned_toks.append(tokenizer.convert_tokens_to_string([tok]).strip())
    return cleaned_toks


def capture_salient(claim, claim_to_compare):
    reduced = claim["reduced_claim_tokens"]
    orig = claim["orig_claim_tokens"]
    orig_to_compare = claim_to_compare["orig_claim_tokens"]

    # Mask reduced tokens in original tokens
    reduced_mask = [0] * len(orig)
    i = j = 0
    assert len(reduced) <= len(orig)
    while i < len(orig) and j < len(reduced):
        if orig[i] == reduced[j]:
            reduced_mask[i] = 1
            i += 1
            j += 1
        else:
            i += 1  # increase original index

    # Mask salient (different) tokens in original tokens
    salient_mask = [0] * len(orig)
    i = j = 0
    while i < len(orig) and j < len(orig_to_compare):
        if orig[i] == orig_to_compare[j]:
            i += 1
            j += 1
        else:
            salient_mask[i] = 1
            if len(orig) - i < len(orig_to_compare) - j:
                j += 1
            elif len(orig) - i > len(orig_to_compare) - j:
                i += 1
            else:
                i += 1
                j += 1

    logger.info(f"     reduced: {reduced}")
    logger.info(f"   comparing: {orig_to_compare}")
    logger.info(f"        orig: {orig}")
    logger.info(f"reduced mask: {reduced_mask}")
    logger.info(f"salient mask: {salient_mask}")

    for reduced, salient in zip(reduced_mask, salient_mask):
        if salient and not reduced:
            return False
    return True


def evalute(claim, stat):
    if claim["label"] == claim["pred_label"]:
        stat["n_correct"] += 1
        if claim["capture"]:
            stat["n_correct_capture"] += 1
    logger.info(f'label & pred_label: {claim["label"]} {claim["pred_label"]}')
    logger.info(
        f'correct & capture: {claim["label"] == claim["pred_label"]} {claim["capture"]}'
    )


def main(dataname, loss, tokenizer, pretrained, split):
    abs_path = Path(__file__).parent.resolve()
    filepath = f"{abs_path}/data/{dataname}/{split}.jsonl"
    original_data = [line for line in jsonlines.open(filepath)]

    if loss == "cross_entropy":
        filepath = (
            f"{abs_path}/{dataname}_{loss}/{pretrained}-128-out/{split}.reduced.jsonl"
        )
    else:
        best_beta_file = f"{abs_path}/{dataname}_{loss}/best_beta.txt"
        with open(best_beta_file) as f:
            best_beta = f.readline().strip()
        filepath = f"{abs_path}/{dataname}_{loss}/{pretrained}-128-{best_beta}-out/{split}.reduced.jsonl"  # noqa: E501
    reduced_data = [line for line in jsonlines.open(filepath)]

    sanity_check(original_data, reduced_data)

    group_by_evidence = defaultdict(lambda: [])
    for original_line, reduced_line in zip(original_data, reduced_data):
        evidence = " ".join(original_line["evidence"])
        label = original_line["label"][0]
        pred_label = reduced_line["pred_label"]
        group_by_evidence[evidence].append(
            {
                "reduced_claim_tokens": clean_tokens(
                    reduced_line["claim_tokens"], tokenizer
                ),
                "orig_claim_tokens": clean_tokens(
                    reduced_line["orig_claim_tokens"], tokenizer
                ),
                "label": label,
                "pred_label": pred_label,
            }
        )

    stat = {"n_correct": 0, "n_correct_capture": 0}
    for evidence, claims in group_by_evidence.items():
        if len(claims) < 2:  # cannot identify difference with one claim
            continue

        logger.info(f"evidence: {evidence}")
        logger.info(f"# of claims: {len(claims)}")

        supported = []
        refuted = []
        # Split by label
        for claim in claims:
            label = claim["label"]
            if label == "S":
                supported.append(claim)
            elif label == "R":
                refuted.append(claim)
            else:
                raise KeyError(label)

        # Should have only one supported claim
        assert len(supported) == 1 and len(refuted) >= 1

        # Compare supported claim with one of refuted claims
        running_id = 0
        logger.info(f"{running_id}")
        claim = supported[0]
        claim_to_compare = refuted[0]
        claim["capture"] = capture_salient(claim, claim_to_compare)
        evalute(claim, stat)
        running_id += 1

        # Compare each refuted claim with supported claim
        claim_to_compare = supported[0]
        for claim in refuted:
            logger.info(f"{running_id}")
            claim["capture"] = capture_salient(claim, claim_to_compare)
            evalute(claim, stat)
            running_id += 1

        assert running_id == len(claims)
        logger.info("--")

    acc = stat["n_correct_capture"] * 100.0 / stat["n_correct"]
    logger.info(
        f'correct: {stat["n_correct"]} | w/ salient: {stat["n_correct_capture"]} | success: {acc:.1f}'  # noqa: E501
    )


def init_logger(filepath):
    """Initialize logger."""
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    logfile = logging.FileHandler("{}.log".format(filepath), "w")
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)


if __name__ == "__main__":
    dataset = "covidfact"
    pretrained = "roberta-base"
    split = "test"
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)

    for loss in [
        "cross_entropy",
        "label_smoothing",
        "confidence_penalty",
        "j_penalty",
        "input_reduction+label_smoothing",
        "input_reduction+confidence_penalty",
        "input_reduction+j_penalty",
    ]:
        logdir = Path("salient_logs")
        logdir.mkdir(parents=True, exist_ok=True)
        init_logger(logdir / f"{split}_{dataset}_{loss}")
        main(
            dataset,
            loss,
            tokenizer,
            pretrained,
            split,
        )
        if logger.hasHandlers():
            logger.handlers.clear()
