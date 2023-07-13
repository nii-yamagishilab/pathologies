# Copyright (c) 2023, Yamagishi Laboratory, National Institute of Informatics
# Author: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.

import argparse
import pytorch_lightning as pl
from datetime import datetime
from pytorch_lightning.utilities import rank_zero_info
from train import FactVerificationTransformer
from predict import get_dataloader
from temperature_scaling import ModelWithTemperature


def build_args():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--checkpoint_file", type=str, required=True)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, default="best_temperature.txt")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--n_bins", type=int, default=20)
    args = parser.parse_args()
    return args


def main():
    t_start = datetime.now()
    args = build_args()

    model = FactVerificationTransformer.load_from_checkpoint(
        checkpoint_path=args.checkpoint_file,
        strict=True if args.strict else False,
    )
    model.freeze()

    scaled_model = ModelWithTemperature(model)
    scaled_model.set_temperature(get_dataloader(model, args), args.n_bins)

    with open(args.out_file, "w") as f:
        f.write(f"{scaled_model.temperature.item()}\n")

    t_delta = datetime.now() - t_start
    rank_zero_info(f"Calibration took '{t_delta}'")


if __name__ == "__main__":
    main()
