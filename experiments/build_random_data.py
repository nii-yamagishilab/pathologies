# Copyright (c) 2023, Yamagishi Laboratory, National Institute of Informatics
# Author: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.
#
# This code contains many hard-coded paths. It assumes that we already
# created reduced examples in the previous steps and three reduced
# files are in the following paths:
#   - data/{dataname}/train.reduced.jsonl
#   - {dataname}_cross_entropy/{pretrained}-128-out/{dev,test}.reduced.jsonl.
# The original files are in:
#   - data/{dataname}/{train,dev,test}.jsonl
# Also, it assumes that we used RoBERTa-base's tokenizer.


import jsonlines
import random
import re
import sys
from pathlib import Path
from transformers import AutoTokenizer
from build_reduced_data import sanity_check


def tokens_to_text(tokens, tokenizer):
    # Convert token-by-token to prevent the tokenizer from incorrectly
    # merging an orphan suffix token to its previous token
    strings = [tokenizer.convert_tokens_to_string([token]) for token in tokens]
    text = " ".join(strings)
    return re.sub(" +", " ", text).strip()


def run(dataname, split, tokenizer, pretrained="roberta-base"):
    abs_path = Path(__file__).parent.resolve()
    original_file = f"{abs_path}/data/{dataname}/{split}.jsonl"
    if split == "train":
        reduced_file = f"{abs_path}/data/{dataname}/{split}.reduced.jsonl"
    else:
        reduced_file = f"{abs_path}/{dataname}_cross_entropy/{pretrained}-128-out/{split}.reduced.jsonl"  # noqa: E501

    original_data = [line for line in jsonlines.open(original_file)]
    reduced_data = [line for line in jsonlines.open(reduced_file)]
    sanity_check(original_data, reduced_data)

    for original_line, reduced_line in zip(original_data, reduced_data):
        reduced_claim_toks = reduced_line["claim_tokens"].split()
        orig_claim_toks = reduced_line["orig_claim_tokens"].split()
        reduced_claim_len = len(reduced_claim_toks)
        orig_claim_len = len(orig_claim_toks)
        assert reduced_claim_len <= orig_claim_len

        # Randomly select original claim tokens with the same length as reduced claim tokens
        rand_claim_tok_ids = random.sample(
            list(range(orig_claim_len)), reduced_claim_len
        )

        claim_toks_to_keep = []
        for i, tok in enumerate(orig_claim_toks):
            if i in rand_claim_tok_ids:
                claim_toks_to_keep.append(tok)
        assert len(claim_toks_to_keep) == reduced_claim_len

        reduced_line["label"] = original_line["label"]
        reduced_line["claim_tokens"] = " ".join(claim_toks_to_keep)
        reduced_line["claim"] = tokens_to_text(claim_toks_to_keep, tokenizer)

    out_dir = Path(f"{abs_path}/data/random-reduced-{dataname}")
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}.jsonl"

    with jsonlines.open(out_path, "w") as f:
        f.write_all(reduced_data)
    print(f"Saved to {out_path}")


def main():
    random.seed(3435)  # for reproducibility

    tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)

    for split in ["train", "dev", "test"]:
        run(sys.argv[1], split, tokenizer)


if __name__ == "__main__":
    main()
