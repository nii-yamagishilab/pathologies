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
import sys
from pathlib import Path
from unidecode import unidecode
from nltk.tokenize import RegexpTokenizer


def sanity_check(original_data, reduced_data):
    assert len(original_data) == len(reduced_data)
    tokenizer = RegexpTokenizer(r"\w+")
    # Check whether orginal_claim in reduced data equals claim in original data
    for original_line, reduced_line in zip(original_data, reduced_data):
        claim1 = " ".join(tokenizer.tokenize(unidecode("NFD", original_line["claim"])))
        claim2 = " ".join(
            tokenizer.tokenize(unidecode("NFD", reduced_line["orig_claim"]))
        )
        assert claim1 == claim2


def run(dataname, split, pretrained="roberta-base"):
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
        reduced_line["label"] = original_line["label"]

    out_dir = Path(f"{abs_path}/data/reduced-{dataname}")
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}.jsonl"

    with jsonlines.open(out_path, "w") as f:
        f.write_all(reduced_data)
    print(f"Saved to {out_path}")


def main():
    for split in ["train", "dev", "test"]:
        run(sys.argv[1], split)


if __name__ == "__main__":
    main()
