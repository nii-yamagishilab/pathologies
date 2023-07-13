# Copyright (c) 2023, Yamagishi Laboratory, National Institute of Informatics
# Author: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.

import re
import numpy as np
import torch
from copy import deepcopy
from torch import nn
from tqdm import tqdm
from pytorch_lightning.utilities import rank_zero_info
from processors import FactVerificationProcessor


class Buffers:
    def __init__(self):
        self.input_ids = []
        self.attention_mask = []
        self.token_type_ids = []
        self.pred_labels = []
        self.running_ids = []

    def add(self, i, running_ids, red_inputs, pred_label):
        self.input_ids.append(red_inputs["input_ids"][i])
        self.attention_mask.append(red_inputs["attention_mask"][i])
        self.token_type_ids.append(red_inputs["token_type_ids"][i])
        self.pred_labels.append(pred_label)
        self.running_ids.append(running_ids[i])


class InputReducer(nn.Module):
    def __init__(self, model):
        super(InputReducer, self).__init__()
        self.hparams = model.hparams
        self.config = model.config
        self.tokenizer = model.tokenizer
        self.ir_model = model

    def tokens_to_text(self, tokens):
        # Convert token-by-token to prevent the tokenizer from incorrectly
        # merging an orphan suffix token to its previous token
        strings = [self.tokenizer.convert_tokens_to_string([token]) for token in tokens]
        text = " ".join(strings)
        return re.sub(" +", " ", text).strip()

    def ids_to_tokens(self, ids):
        ids = [i for i in ids if i not in set(self.tokenizer.all_special_ids)]
        return self.tokenizer.convert_ids_to_tokens(ids)

    def split_input_ids(self, input_ids):
        sep_id = input_ids.tolist().index(self.tokenizer.sep_token_id)
        return input_ids[:sep_id], input_ids[sep_id:]

    def convert_ids_to_tokens(self, fin_inputs, inputs, id2label):
        examples = []
        for (
            fin_input_ids,
            fin_conf,
            fin_pred_label,
            orig_input_ids,
            orig_conf,
        ) in zip(
            fin_inputs["input_ids"].cpu().numpy(),
            fin_inputs["confs"],
            fin_inputs["pred_labels"],
            inputs["input_ids"].cpu().numpy(),
            inputs["confs"],
        ):
            fin_claim_ids, fin_evidence_ids = self.split_input_ids(fin_input_ids)
            orig_claim_ids, orig_evidence_ids = self.split_input_ids(orig_input_ids)

            fin_evidence_tokens = self.ids_to_tokens(fin_evidence_ids)
            orig_evidence_tokens = self.ids_to_tokens(orig_evidence_ids)

            fin_evidence_text = self.tokens_to_text(fin_evidence_tokens)
            orig_evidence_text = self.tokens_to_text(orig_evidence_tokens)

            assert (
                fin_evidence_text == orig_evidence_text
            ), f"{fin_evidence_text} != {orig_evidence_text}"

            orig_claim_tokens = self.ids_to_tokens(orig_claim_ids)
            fin_claim_tokens = self.ids_to_tokens(fin_claim_ids)

            orig_claim_text = self.tokens_to_text(orig_claim_tokens)
            fin_claim_text = self.tokens_to_text(fin_claim_tokens)

            examples.append(
                {
                    "evidence": orig_evidence_text,
                    "evidence_tokens": " ".join(orig_evidence_tokens),
                    "orig_claim": orig_claim_text,
                    "orig_claim_tokens": " ".join(orig_claim_tokens),
                    "orig_conf": f"{orig_conf:.3f}",
                    "claim": fin_claim_text,
                    "claim_tokens": " ".join(fin_claim_tokens),
                    "pred_label": id2label[fin_pred_label],
                    "conf": f"{fin_conf:.3f}",
                }
            )
        return examples

    def print_recduction_path(self, inputs):
        if self.hparams.print_reduction_path and inputs["input_ids"].shape[0] == 1:
            conf = inputs["confs"][0]
            claim_ids, _ = self.split_input_ids(inputs["input_ids"][0].cpu().numpy())
            claim_tokens = self.ids_to_tokens(claim_ids)
            claim_text = self.tokens_to_text(claim_tokens)
            rank_zero_info(f"({conf:.3f}) {claim_text}")

    def create_token_type_ids(self, inputs):
        token_type_ids_list = []
        for input_ids in inputs["input_ids"].cpu().tolist():
            token_ids_0_len = input_ids.index(self.tokenizer.sep_token_id)
            token_ids_1_len = self.hparams.max_seq_length - token_ids_0_len
            token_type_ids = token_ids_0_len * [0] + token_ids_1_len * [1]
            assert len(token_type_ids) == len(input_ids)
            token_type_ids_list.append(token_type_ids)
        return torch.as_tensor(np.stack(token_type_ids_list)).cuda()

    def get_claim_len(self, inputs):
        mask = (
            (inputs["token_type_ids"] == 0)
            & (inputs["input_ids"] != self.tokenizer.bos_token_id)
            & (inputs["input_ids"] != self.tokenizer.eos_token_id)
            & (inputs["input_ids"] != self.tokenizer.cls_token_id)
            & (inputs["input_ids"] != self.tokenizer.sep_token_id)
            & (inputs["input_ids"] != self.tokenizer.pad_token_id)
        )  # claim mask
        return mask.sum(axis=-1)

    def update_final(self, i, running_ids, fin_inputs, red_inputs, red_confs):
        fin_inputs["input_ids"][running_ids[i]] = red_inputs["input_ids"][i]
        fin_inputs["attention_mask"][running_ids[i]] = red_inputs["attention_mask"][i]
        fin_inputs["token_type_ids"][running_ids[i]] = red_inputs["token_type_ids"][i]
        fin_inputs["claim_len"][running_ids[i]] = red_inputs["claim_len"][i]
        fin_inputs["confs"][running_ids[i]] = red_confs[i]

    def copy_buffers_to_inputs(self, inputs, buffers):
        inputs["input_ids"] = torch.vstack(buffers.input_ids)
        inputs["attention_mask"] = torch.vstack(buffers.attention_mask)
        inputs["token_type_ids"] = torch.vstack(buffers.token_type_ids)
        inputs["pred_labels"] = buffers.pred_labels

    def get_pred_labels(self, logits):
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        pred_labels = np.argmax(probs, axis=1).tolist()
        confs = np.max(probs, axis=1).tolist()
        return pred_labels, confs

    def get_grads_norm(self, inputs):
        embeddings = getattr(
            self.ir_model.model, self.config.model_type
        ).get_input_embeddings()
        copied_inputs = deepcopy(inputs)
        inputs_embeds = embeddings(copied_inputs["input_ids"])
        copied_inputs["inputs_embeds"] = inputs_embeds
        del copied_inputs["input_ids"]
        copied_inputs["labels"] = torch.as_tensor(inputs["pred_labels"]).cuda()
        outputs = self.ir_model(**copied_inputs)
        grads = torch.autograd.grad(
            outputs.loss,
            inputs_embeds,
        )[0]
        grads_norm = torch.norm(grads, dim=-1)
        mask = (
            (inputs["token_type_ids"] == 0)
            & (inputs["input_ids"] != self.tokenizer.bos_token_id)
            & (inputs["input_ids"] != self.tokenizer.eos_token_id)
            & (inputs["input_ids"] != self.tokenizer.cls_token_id)
            & (inputs["input_ids"] != self.tokenizer.sep_token_id)
            & (inputs["input_ids"] != self.tokenizer.pad_token_id)
        )  # claim mask
        grads_norm = grads_norm.masked_fill(~mask, float("inf"))
        return grads_norm.cpu().numpy()

    def remove_one_token(self, inputs):
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []
        claim_len_list = []
        pad_token_id = self.tokenizer.pad_token_id
        grads_norm = self.get_grads_norm(inputs)
        claim_lengths = np.sum(grads_norm != float("inf"), axis=-1)
        for grad_norm, input_ids, attention_mask, token_type_ids, claim_len in zip(
            grads_norm,
            inputs["input_ids"].cpu().tolist(),
            inputs["attention_mask"].cpu().tolist(),
            inputs["token_type_ids"].cpu().tolist(),
            claim_lengths,
        ):
            smallest = np.argmin(grad_norm)
            # Skip and pad
            input_ids_list.append(
                input_ids[:smallest] + input_ids[smallest + 1 :] + [pad_token_id]
            )
            attention_mask_list.append(
                attention_mask[:smallest] + attention_mask[smallest + 1 :] + [0]
            )
            token_type_ids_list.append(
                token_type_ids[:smallest] + token_type_ids[smallest + 1 :] + [1]
            )
            claim_len_list.append(claim_len - 1)

        inputs["input_ids"] = torch.as_tensor(np.stack(input_ids_list)).cuda()
        inputs["attention_mask"] = torch.as_tensor(np.stack(attention_mask_list)).cuda()
        inputs["token_type_ids"] = torch.as_tensor(np.stack(token_type_ids_list)).cuda()
        inputs["claim_len"] = torch.as_tensor(np.stack(claim_len_list)).cuda()
        return inputs

    def input_reduction_batch(self, inputs):
        fin_inputs = deepcopy(inputs)
        running_ids = [i for i in range(len(inputs["input_ids"]))]
        self.print_recduction_path(fin_inputs)

        while True:
            red_inputs = self.remove_one_token(inputs)
            red_outputs = self.ir_model(**red_inputs)
            red_pred_labels, red_confs = self.get_pred_labels(red_outputs.logits)
            buffers = Buffers()
            for i, (red_pred_label, pred_label) in enumerate(
                zip(red_pred_labels, inputs["pred_labels"])
            ):
                if red_pred_label == pred_label and red_inputs["claim_len"][i] > 0:
                    self.update_final(i, running_ids, fin_inputs, red_inputs, red_confs)
                    buffers.add(i, running_ids, red_inputs, pred_label)
                    self.print_recduction_path(fin_inputs)

            if len(buffers.input_ids):
                self.copy_buffers_to_inputs(inputs, buffers)
            else:
                break
            running_ids = buffers.running_ids

        return fin_inputs

    def input_reduction(self, dataloader):
        self.eval().cuda()
        id2label = {
            i: label for i, label in enumerate(FactVerificationProcessor().get_labels())
        }
        num_batches = len(dataloader)
        fin_data = []
        for batch in tqdm(dataloader, total=num_batches, desc="Reduce"):
            batch = [b.cuda() for b in batch]
            inputs = self.ir_model.build_inputs(batch)
            outputs = self.ir_model(**inputs)
            pred_labels, confs = self.get_pred_labels(outputs.logits)
            inputs["pred_labels"] = pred_labels
            inputs["confs"] = confs
            if inputs["token_type_ids"] is None:
                inputs["token_type_ids"] = self.create_token_type_ids(inputs)
            inputs["claim_len"] = self.get_claim_len(inputs)
            if "labels" in inputs:
                del inputs["labels"]  # omit ground-truth labels
            fin_inputs = self.input_reduction_batch(deepcopy(inputs))
            fin_data.extend(self.convert_ids_to_tokens(fin_inputs, inputs, id2label))
        return fin_data
