# Copyright (c) 2023, Yamagishi Laboratory, National Institute of Informatics
# Author: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.

import math
import warnings
import pytorch_lightning as pl
from datetime import datetime
from pytorch_lightning.utilities import rank_zero_info
from transformers import AutoTokenizer, logging
from transformers.optimization import (
    AdamW,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from adafactor import Adafactor

warnings.filterwarnings("ignore")
logging.set_verbosity_error()

arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
}


class BaseTransformer(pl.LightningModule):
    def __init__(
        self,
        hparams,
        model,
        **config_kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.config = model.config
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.pretrained_model_name, use_fast=True
        )
        self.model = model

    def setup(self, stage):
        if self.training and stage == "fit":
            self.init_parameters()

            self.train_loader = self.get_dataloader(
                "train",
                self.hparams.train_batch_size,
                self.hparams.num_workers,
            )

            effective_batch_size = (
                self.hparams.train_batch_size
                * self.hparams.accumulate_grad_batches
                * max(1, self.hparams.gpus)
            )

            dataset_size = len(self.train_loader.dataset)

            self.total_steps = (
                dataset_size / effective_batch_size
            ) * self.hparams.max_epochs
        else:
            self.total_steps = 0

    def get_dataloader(self, mode, batch_size, num_workers):
        raise NotImplementedError("You must implement this for your task")

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.get_dataloader(
            "dev", self.hparams.eval_batch_size, self.hparams.num_workers
        )

    def test_dataloader(self):
        return self.get_dataloader(
            "test", self.hparams.eval_batch_size, self.hparams.num_workers
        )

    def get_lr_scheduler(self):
        get_schedule_func = arg_to_scheduler[self.hparams.lr_scheduler]
        if self.hparams.warmup_steps > 0:
            rank_zero_info(f"warmup_steps: {self.hparams.warmup_steps:.2f}")

        if self.hparams.warmup_ratio > 0 and self.hparams.warmup_steps == 0:
            assert self.hparams.warmup_ratio > 0 and self.hparams.warmup_ratio <= 1
            self.hparams.warmup_steps = math.ceil(
                self.total_steps * self.hparams.warmup_ratio
            )
            rank_zero_info(f"total_steps: {self.total_steps:.2f}")
            rank_zero_info(f"warmup_ratio: {self.hparams.warmup_ratio:.2f}")
            rank_zero_info(f"warmup_steps: {self.hparams.warmup_steps:.2f}")

        self.lr_scheduler = get_schedule_func(
            self.opt,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": self.lr_scheduler, "interval": "step", "frequency": 1}
        return scheduler

    def configure_optimizers(self):
        model = self.model
        for n, p in model.named_parameters():
            if any(f in n for f in self.hparams.freeze):
                p.requires_grad_(False)
                rank_zero_info(f"Freeze [{n}]")

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        if self.hparams.adafactor:
            optimizer = Adafactor(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                scale_parameter=False,
                relative_step=False,
            )
        else:
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                eps=self.hparams.adam_epsilon,
            )
        self.opt = optimizer
        scheduler = self.get_lr_scheduler()
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--data_dir", type=str, required=True)
        parser.add_argument("--model_name", type=str, default="base")
        parser.add_argument(
            "--pretrained_model_name", type=str, default="bert-base-uncased"
        )
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--lr_scheduler", type=str, default="linear")
        parser.add_argument("--min_lr", type=float, default=1e-5)
        parser.add_argument("--weight_decay", type=float, default=0.0)
        parser.add_argument("--adam_epsilon", type=float, default=1e-8)
        parser.add_argument("--warmup_ratio", type=float, default=0.0)
        parser.add_argument("--warmup_steps", type=int, default=0)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--train_batch_size", type=int, default=32)
        parser.add_argument("--eval_batch_size", type=int, default=32)
        parser.add_argument("--adafactor", action="store_true")
        parser.add_argument("--seed", type=int, default=3435)
        parser.add_argument("--patience", type=int, default=3)
        parser.add_argument("--skip_validation", action="store_true")
        parser.add_argument("--freeze", nargs="+", default=[])


def generic_train(model, args, callbacks):
    train_params = {}

    if args.gpus > 1:
        train_params["distributed_backend"] = "ddp"

    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        **train_params,
    )

    t_start = datetime.now()
    trainer.fit(model)
    rank_zero_info(f"\nTraining took '{datetime.now() - t_start}'")

    return trainer
