# Copyright (c) 2023, Yamagishi Laboratory, National Institute of Informatics
# Author: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.

import argparse
import numpy as np
import pytorch_lightning as pl
from datetime import datetime
from pathlib import Path
from pytorch_lightning.utilities import rank_zero_info
from torch.utils.data import TensorDataset, DataLoader
from train import FactVerificationTransformer


def get_dataloader(model, args):
    filepath = Path(args.in_file)
    assert filepath.exists(), f"Cannot find [{filepath}]"
    dataset_type = filepath.stem
    feature_list = model.create_features(dataset_type, filepath)
    return DataLoader(
        TensorDataset(*feature_list),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )


def build_args():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--checkpoint_file", type=str, required=True)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_penultimate_layer", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--use_reduced_inputs", action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = build_args()

    assert args.checkpoint_file
    model = FactVerificationTransformer.load_from_checkpoint(
        checkpoint_path=args.checkpoint_file,
        strict=True if args.strict else False,
    )
    model.freeze()

    params = {}
    params["precision"] = model.hparams.precision
    trainer = pl.Trainer.from_argparse_args(args, logger=False, **params)

    model.hparams.save_penultimate_layer = args.save_penultimate_layer
    model.hparams.temperature = args.temperature
    model.hparams.use_reduced_inputs = args.use_reduced_inputs

    t_start = datetime.now()
    predictions = trainer.predict(model, get_dataloader(model, args))
    t_delta = datetime.now() - t_start
    rank_zero_info(f"Prediction took '{t_delta}'")

    probs, embs = [], []
    for p in predictions:
        probs.append(p.probs)
        if p.embs is not None:
            embs.append(p.embs)

    probs = np.vstack(probs)

    if args.out_file:
        filepath = Path(args.out_file)
    else:
        filepath = Path(args.in_file).with_suffix(".prob")


    np.savetxt(filepath, probs, delimiter=" ", fmt="%.5f")
    rank_zero_info(f"Saved output probabilities to {filepath}")

    if embs:
        embs = np.vstack(embs)
        filepath = filepath.with_suffix(".emb.npy")
        np.save(filepath, embs)
        rank_zero_info(f"Saved penultimate_layer to {filepath}")


if __name__ == "__main__":
    main()
