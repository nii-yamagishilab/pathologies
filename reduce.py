# Copyright (c) 2023, Yamagishi Laboratory, National Institute of Informatics
# Author: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.

import argparse
import jsonlines
import pytorch_lightning as pl
from datetime import datetime
from pathlib import Path
from pytorch_lightning.utilities import rank_zero_info
from train import FactVerificationTransformer
from predict import get_dataloader
from input_reduction import InputReducer


def build_args():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--checkpoint_file", type=str, required=True)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--print_reduction_path", action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = build_args()

    if args.print_reduction_path:
        assert args.batch_size == 1, "Support --print_reduction_path with --batch_size 1 only"

    assert args.checkpoint_file
    model = FactVerificationTransformer.load_from_checkpoint(
        checkpoint_path=args.checkpoint_file,
        strict=True if args.strict else False,
    )

    model.hparams.print_reduction_path = args.print_reduction_path

    reducer = InputReducer(model)
    t_start = datetime.now()
    reduced_data = reducer.input_reduction(get_dataloader(model, args))
    t_delta = datetime.now() - t_start
    rank_zero_info(f"Input reduction took '{t_delta}'")

    if args.out_file:
        filepath = Path(args.out_file)
    else:
        filepath = Path(args.in_file).with_suffix(".reduced.jsonl")

    with jsonlines.open(filepath, "w") as out:
        out.write_all(reduced_data)
    print(f"Saved reduced inputs to {filepath}")


if __name__ == "__main__":
    main()
