# Copyright (c) 2023, Yamagishi Laboratory, National Institute of Informatics
# Author: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.

import torch
from torch import nn
from transformers import AutoConfig, AutoModel
from transformers.modeling_utils import PreTrainedModel
from transformers.file_utils import ModelOutput
from criterions import (
    cross_entropy,
    label_smoothing,
    confidence_penalty,
    j_penalty,
    js_penalty,
)

LOSS_FCT = {
    "cross_entropy": cross_entropy,
    "label_smoothing": label_smoothing,
    "confidence_penalty": confidence_penalty,
    "j_penalty": j_penalty,
    "js_penalty": js_penalty,
}


class BaseModelOutput(ModelOutput):
    loss: torch.FloatTensor = None
    ce_loss: torch.FloatTensor = None
    reg_loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    penultimate_layer: torch.FloatTensor = None


class Classifier(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_labels,
        dropout=0.1,
    ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        z = torch.relu(x)
        x = self.dropout(z)
        x = self.out_proj(x)
        return x, z


class BaseModel(PreTrainedModel):
    def __init__(self, hparams, num_labels):
        config = AutoConfig.from_pretrained(
            hparams.pretrained_model_name, num_labels=num_labels
        )
        super().__init__(config)
        setattr(
            self,
            self.config.model_type,
            AutoModel.from_pretrained(
                hparams.pretrained_model_name, config=self.config
            ),
        )
        self.classifier = Classifier(
            self.config.hidden_size,
            num_labels,
            dropout=hparams.classifier_dropout_prob,
        )
        self.hparams = hparams

    def forward(
        self,
        input_ids=None,
        labels=None,
        **kwargs,
    ):
        model = getattr(self, self.config.model_type)

        model_args_name = set(model.forward.__code__.co_varnames[1:])  # skip self
        if self.config.model_type == "roberta" and "token_type_ids" in model_args_name:
            model_args_name.remove("token_type_ids")
        valid_kwargs = {
            key: value for (key, value) in kwargs.items() if key in model_args_name
        }

        encoder_outputs = model(input_ids, **valid_kwargs)
        features = encoder_outputs.last_hidden_state[:, 0]  # equiv. to [CLS]
        logits, penultimate_layer = self.classifier(features)

        loss, ce_loss, reg_loss = None, None, None
        if labels is not None:
            if (
                hasattr(self.hparams, "use_reduced_inputs")
                and self.hparams.use_reduced_inputs
                and "reduced_input_ids" in kwargs
            ):
                reduced_encoder_outputs = model(
                    input_ids=kwargs["reduced_input_ids"],
                    attention_mask=kwargs["reduced_attention_mask"],
                    token_type_ids=kwargs["reduced_token_type_ids"],
                )
                reduced_features = reduced_encoder_outputs.last_hidden_state[:, 0]
                reduced_logits, _ = self.classifier(reduced_features)
                loss, ce_loss, reg_loss = LOSS_FCT[self.hparams.loss_function](
                    logits.view(-1, self.config.num_labels),
                    labels.view(-1),
                    beta=self.hparams.beta,
                    reduced_logits=reduced_logits.view(-1, self.config.num_labels),
                )
            else:
                loss, ce_loss, reg_loss = LOSS_FCT[self.hparams.loss_function](
                    logits.view(-1, self.config.num_labels),
                    labels.view(-1),
                    beta=self.hparams.beta,
                )

        return BaseModelOutput(
            loss=loss,
            ce_loss=ce_loss,
            reg_loss=reg_loss,
            logits=logits,
            penultimate_layer=penultimate_layer,
        )
