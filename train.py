# Copyright (c) 2023, Yamagishi Laboratory, National Institute of Informatics
# Author: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.

import argparse
import numpy as np
import pytorch_lightning as pl
import torch
from argparse import Namespace
from filelock import FileLock
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from transformers.file_utils import ModelOutput
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_info
from lightning_base import BaseTransformer, generic_train
from modeling_base import BaseModel
from processors import (
    FactVerificationProcessor,
    compute_metrics,
    convert_examples_to_features,
)


class PredictionOutput(ModelOutput):
    probs: np.ndarray = None
    embs: np.ndarray = None


class FactVerificationTransformer(BaseTransformer):
    def __init__(self, hparams, **kwargs):
        if type(hparams) == dict:
            hparams = Namespace(**hparams)
        labels = FactVerificationProcessor().get_labels(
            Path(hparams.data_dir) / "train.jsonl"
        )
        num_labels = len(labels)
        rank_zero_info(f"num_labels: {num_labels}")
        model = BaseModel(hparams, num_labels)
        super().__init__(hparams, model)

    def create_features(self, set_type, filepath):
        rank_zero_info(f"Create features from: {filepath}")
        hparams = self.hparams
        processor = FactVerificationProcessor()

        _use_reduced_inputs = False
        if "reduced" in str(filepath):
            if (
                hasattr(hparams, "use_reduced_inputs") and hparams.use_reduced_inputs
            ) or (
                hasattr(hparams, "train_on_reduced_inputs")
                and hparams.train_on_reduced_inputs
            ):
                # If _use_reduced_inputs is True, processor will
                # directly convert claim's tokens to ids without using
                # tokenizer.encode() to avoid re-tokenizing twice.
                _use_reduced_inputs = True
                rank_zero_info(
                    f"\U0001F4A5 _use_reduced_inputs = {_use_reduced_inputs} for {filepath}"
                )

        examples = processor.get_examples(
            filepath,
            set_type,
            self.training,
            hparams.use_title,
            hparams.claim_only,
            _use_reduced_inputs,
        )
        num_examples = len(examples)
        features = convert_examples_to_features(
            examples,
            self.tokenizer,
            processor.get_labels(),
            hparams.max_seq_length,
            hparams.num_workers,
            _use_reduced_inputs,
        )

        def empty_tensor_1():
            return torch.empty(num_examples, dtype=torch.long)

        def empty_tensor_2():
            return torch.empty((num_examples, hparams.max_seq_length), dtype=torch.long)

        input_ids = empty_tensor_2()
        attention_mask = empty_tensor_2()
        token_type_ids = empty_tensor_2()
        labels = empty_tensor_1()
        for i, feature in enumerate(features):
            input_ids[i] = torch.tensor(feature.input_ids)
            attention_mask[i] = torch.tensor(feature.attention_mask)
            if feature.token_type_ids is not None:
                token_type_ids[i] = torch.tensor(feature.token_type_ids)
            labels[i] = torch.tensor(feature.label)
        return [input_ids, attention_mask, token_type_ids, labels]

    def cached_feature_file(self, mode):
        dirname = "pathologies_" + Path(self.hparams.data_dir).parts[-1]
        feat_dirpath = Path(self.hparams.cache_dir) / dirname
        feat_dirpath.mkdir(parents=True, exist_ok=True)
        pt = self.hparams.pretrained_model_name.replace("/", "__")
        return (
            feat_dirpath
            / f"cached_{mode}_{pt}_{self.hparams.max_seq_length}_{self.hparams.seed}"
        )

    def prepare_data(self):
        if self.training:
            dataset_types = ["train", "dev"]
            for dataset_type in dataset_types:
                if dataset_type == "dev" and self.hparams.skip_validation:
                    continue
                cached_feature_file = self.cached_feature_file(dataset_type)
                lock_path = cached_feature_file.with_suffix(".lock")
                with FileLock(lock_path):
                    if (
                        cached_feature_file.exists()
                        and not self.hparams.overwrite_cache
                    ):
                        rank_zero_info(f"Feature file [{cached_feature_file}] exists!")
                        continue
                    filepath = Path(self.hparams.data_dir) / f"{dataset_type}.jsonl"
                    assert filepath.exists(), f"Cannot find: {filepath}"
                    feature_list = self.create_features(dataset_type, filepath)

                    # Use reduced examples for regularization
                    if dataset_type == "train" and self.hparams.use_reduced_inputs:
                        assert (
                            not self.hparams.train_on_reduced_inputs
                        ), "Cannot train and regularize with reduced examples at the same time."

                        # Assume 'train.reduced.jsonl' is already
                        # created and located in the same directory as
                        # 'train.jsonl'
                        data_dir = Path(self.hparams.data_dir)
                        filepath = data_dir / f"{dataset_type}.reduced.jsonl"
                        assert filepath.exists(), f"Cannot find: {filepath}"
                        reduced_feature_list = self.create_features(
                            dataset_type, filepath
                        )
                        feature_list.extend(reduced_feature_list)

                    torch.save(feature_list, cached_feature_file)
                    rank_zero_info(f"\u2728 Saved features to: {cached_feature_file}")

    def init_parameters(self):
        base_name = self.config.model_type
        no_init = [base_name]
        rank_zero_info(f"\U0001F4A5 Force no_init to: {base_name}")
        if self.hparams.no_init:
            no_init.extend(self.hparams.no_init)
            rank_zero_info(f"\U0001F4A5 Force no_init to: {self.hparams.no_init}")
        for n, p in self.model.named_parameters():
            if any(ni in n for ni in no_init):
                continue
            rank_zero_info(f"Initialize: {n}")
            if "bias" not in n:
                if hasattr(self.config, "initializer_range"):
                    p.data.normal_(mean=0.0, std=self.config.initializer_range)
                else:
                    p.data.normal_(mean=0.0, std=0.02)
            else:
                p.data.zero_()

    def get_dataloader(self, mode, batch_size, num_workers):
        if self.training and mode == "dev" and self.hparams.skip_validation:
            return None
        cached_feature_file = self.cached_feature_file(mode)
        assert cached_feature_file.exists(), f"Cannot find: {cached_feature_file}"
        feature_list = torch.load(cached_feature_file)
        shuffle = True if "train" in mode and self.training else False
        rank_zero_info(
            f"Load features from [{cached_feature_file}] with shuffle={shuffle}"
        )
        return DataLoader(
            TensorDataset(*feature_list),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    def forward(self, **inputs):
        return self.model(**inputs)

    def build_inputs(self, batch):
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": None if self.config.model_type == "roberta" else batch[2],
            "labels": batch[3],
        }
        if (
            hasattr(self.hparams, "use_reduced_inputs")
            and self.hparams.use_reduced_inputs
            and len(batch) == 4 * 2
        ):
            inputs["reduced_input_ids"] = batch[4]
            inputs["reduced_attention_mask"] = batch[5]
            inputs["reduced_token_type_ids"] = (
                None if self.config.model_type == "roberta" else batch[6]
            )
        return inputs

    def base_training_step(self, inputs, batch_idx):
        outputs = self(**inputs)
        log_dict = {
            "train_loss": outputs.loss.detach().cpu(),
            "lr": self.lr_scheduler.get_last_lr()[-1],
        }
        if outputs.ce_loss is not None:
            log_dict["ce_loss"] = outputs.ce_loss.detach().cpu()
        if outputs.reg_loss is not None:
            log_dict["reg_loss"] = outputs.reg_loss.detach().cpu()
        self.log_dict(log_dict)
        return outputs.loss

    def training_step(self, batch, batch_idx):
        inputs = self.build_inputs(batch)
        return self.base_training_step(inputs, batch_idx)

    def validation_step(self, batch, batch_idx):
        inputs = self.build_inputs(batch)
        outputs = self(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        return {
            "loss": outputs.loss.detach().cpu(),
            "probs": probs.detach().cpu().numpy(),
            "labels": inputs["labels"].detach().cpu().numpy(),
        }

    def predict_step(self, batch, batch_idx):
        inputs = self.build_inputs(batch)
        outputs = self(**inputs)
        logits = outputs.logits / self.hparams.temperature
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        embs = None
        if self.hparams.save_penultimate_layer:
            embs = outputs.penultimate_layer.detach().cpu().numpy()
        return PredictionOutput(probs=probs, embs=embs)

    def validation_epoch_end(self, outputs):
        avg_loss = (
            torch.stack([x["loss"] for x in outputs]).mean().detach().cpu().item()
        )
        labels = np.concatenate([x["labels"] for x in outputs], axis=0)
        probs = np.concatenate([x["probs"] for x in outputs], axis=0)
        results = {
            **{"loss": avg_loss},
            **compute_metrics(probs, labels),
        }
        self.log_dict({f"val_{k}": torch.tensor(v) for k, v in results.items()})

    @staticmethod
    def add_model_specific_args(parser):
        BaseTransformer.add_model_specific_args(parser)
        parser.add_argument("--cache_dir", type=str, default="/tmp")
        parser.add_argument("--overwrite_cache", action="store_true")
        parser.add_argument("--save_all_checkpoints", action="store_true")
        parser.add_argument("--max_seq_length", type=int, default=128)
        parser.add_argument("--use_title", action="store_true")
        parser.add_argument("--claim_only", action="store_true")
        parser.add_argument("--no_init", nargs="+", default=[])
        parser.add_argument("--classifier_dropout_prob", type=float, default=0.1)
        parser.add_argument(
            "--loss_function",
            default="cross_entropy",
            choices=[
                "cross_entropy",
                "label_smoothing",
                "confidence_penalty",
                "j_penalty",
                "js_penalty",
            ],
        )
        parser.add_argument("--beta", type=float, default=0.0)
        parser.add_argument("--use_reduced_inputs", action="store_true")
        parser.add_argument("--train_on_reduced_inputs", action="store_true")
        return parser


def build_args():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = FactVerificationTransformer.add_model_specific_args(parser)
    return parser.parse_args()


def main():
    args = build_args()

    if args.seed > 0:
        pl.seed_everything(args.seed)

    model = FactVerificationTransformer(args)

    ckpt_dirpath = Path(args.default_root_dir) / "checkpoints"
    ckpt_dirpath.mkdir(parents=True, exist_ok=True)

    monitor, mode, ckpt_filename = None, "min", "{epoch}-{step}"
    dev_filepath = Path(args.data_dir) / "dev.jsonl"
    if dev_filepath.exists() and not args.skip_validation:
        monitor, mode = "val_acc", "max"
        ckpt_filename = "{epoch}-{step}-{" + monitor + ":.4f}"

    callbacks = []
    callbacks.append(
        ModelCheckpoint(
            dirpath=ckpt_dirpath,
            filename=ckpt_filename,
            monitor=monitor,
            mode=mode,
            save_top_k=-1 if args.save_all_checkpoints else 1,
        )
    )

    if monitor is not None:
        callbacks.append(
            EarlyStopping(monitor=monitor, mode=mode, patience=args.patience)
        )

    generic_train(model, args, callbacks)


if __name__ == "__main__":
    main()
