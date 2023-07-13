# Copyright (c) 2023, Yamagishi Laboratory, National Institute of Informatics
# Author: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.

import jsonlines
import re
import unicodedata
import numpy as np
from transformers import InputExample, InputFeatures, DataProcessor
from functools import partial
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

tokenizer = None


def convert_example_to_features(
    example,
    max_length,
    label2id,
    use_reduced_inputs,
):
    if max_length is None:
        max_length = tokenizer.max_len

    if use_reduced_inputs:
        # claim tokens: example.text_a
        # evidence token: example.text_a
        tokens = example.text_a.split()
        pair_tokens = example.text_b.split()

        # Directly convert tokens to ids
        ids = tokenizer.convert_tokens_to_ids(tokens)
        pair_ids = tokenizer.convert_tokens_to_ids(pair_tokens)

        ids = [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id]
        pair_ids = [tokenizer.bos_token_id] + pair_ids + [tokenizer.eos_token_id]

        len_ids = len(ids)
        len_pair_ids = len(pair_ids)
        total_len = len_ids + len_pair_ids
        num_tokens_to_remove = total_len - max_length

        if num_tokens_to_remove > 0:
            # Truncate on evidence only
            if len_pair_ids > num_tokens_to_remove:
                pair_ids = pair_ids[:-num_tokens_to_remove]
            else:
                raise RuntimeError(
                    f"We need to remove {num_tokens_to_remove} to truncate the input "
                    f"but the second sequence has a length {len(pair_ids)}. "
                )

        # Prepare evidence to combine
        if pair_ids[0] == tokenizer.bos_token_id:
            pair_ids[0] = tokenizer.sep_token_id
        if pair_ids[-1] != tokenizer.eos_token_id:
            pair_ids[-1] = tokenizer.eos_token_id

        # Combine claim and evidence ids
        required_input = ids + pair_ids

        # Create attention mask
        attention_mask = [1] * len(required_input)

        # Pad if necessary
        needs_to_be_padded = len(required_input) != max_length
        if needs_to_be_padded:
            difference = max_length - len(required_input)
            attention_mask = attention_mask + [0] * difference
            required_input = required_input + [tokenizer.pad_token_id] * difference

        # Create input dictionary
        inputs = {"input_ids": required_input, "attention_mask": attention_mask}
    else:
        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            truncation_strategy="only_second",
        )

    label = label2id[example.label]
    return InputFeatures(
        **inputs,
        label=label,
    )


def convert_example_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert


def convert_examples_to_features(
    examples,
    tokenizer,
    label_list,
    max_length=None,
    threads=8,
    use_reduced_inputs=False,
):
    label2id = {label: i for i, label in enumerate(label_list)}
    features = []

    threads = min(threads, cpu_count())
    with Pool(
        threads, initializer=convert_example_to_features_init, initargs=(tokenizer,)
    ) as p:
        annotate_ = partial(
            convert_example_to_features,
            max_length=max_length,
            label2id=label2id,
            use_reduced_inputs=use_reduced_inputs,
        )
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
            )
        )
    return features


def compute_metrics(probs, gold_labels):
    assert len(probs) == len(gold_labels)
    pred_labels = np.argmax(probs, axis=1)
    return {"acc": (gold_labels == pred_labels).mean()}


def process_claim(text):
    text = unicodedata.normalize("NFD", text)
    text = re.sub(r" \-LSB\-.*?\-RSB\-", "", text)
    text = re.sub(r"\-LRB\- \-RRB\- ", "", text)
    text = re.sub(" -LRB-", " ( ", text)
    text = re.sub("-RRB-", " )", text)
    text = re.sub("--", "-", text)
    text = re.sub("``", '"', text)
    text = re.sub("''", '"', text)
    return text


def process_title(text):
    text = unicodedata.normalize("NFD", text)
    text = re.sub("_", " ", text)
    text = re.sub(" -LRB-", " ( ", text)
    text = re.sub("-RRB-", " )", text)
    text = re.sub("-COLON-", ":", text)
    return text


def process_evidence(text):
    text = unicodedata.normalize("NFD", text)
    text = re.sub(" -LSB-.*-RSB-", " ", text)
    text = re.sub(" -LRB- -RRB- ", " ", text)
    text = re.sub("-LRB-", "(", text)
    text = re.sub("-RRB-", ")", text)
    text = re.sub("-COLON-", ":", text)
    text = re.sub("_", " ", text)
    text = re.sub(r"\( *\,? *\)", "", text)
    text = re.sub(r"\( *[;,]", "(", text)
    text = re.sub("--", "-", text)
    text = re.sub("``", '"', text)
    text = re.sub("''", '"', text)
    return text


class FactVerificationProcessor(DataProcessor):
    def get_labels(self, filepath=None):
        labels = ["S", "R", "N"]  # SUPPORTS, REFUTES, NOT ENOUGH INFO
        if filepath:
            if filepath.exists():
                # Check actual unique labels in the data file
                labels_ = set(line["label"][0] for line in jsonlines.open(filepath))
                assert labels_.issubset(set(labels))
                # Remove N label if it does not appear
                if "N" not in labels_:  # SUPPORTS and REFUTES only
                    labels.remove("N")
                return labels
            else:
                # Because filepath is created from
                # [Path(hparams.data_dir) / "train.jsonl"],
                # the dataset name still remains in the checkpoint's hparams.
                if "fever" in str(filepath):
                    return labels
                else:
                    return labels[:-1]  # covidfact and vitaminc
        return labels

    def get_dummy_label(self):
        return "R"

    def get_examples(
        self,
        filepath,
        set_type,
        training=True,
        use_title=True,
        claim_only=False,
        use_reduced_inputs=False,
    ):
        examples = []
        for (i, line) in enumerate(jsonlines.open(filepath)):
            guid = f"{set_type}-{i}"

            if (
                use_reduced_inputs
                and "claim_tokens" in line
                and "evidence_tokens" in line
            ):
                # Keep segmented tokens untouched
                claim = line["claim_tokens"]
                evidence = line["evidence_tokens"]
            else:
                # Clean special symbols
                claim = process_claim(line["claim"])
                evidence = line["evidence"]
                if isinstance(evidence, list):
                    evidence = " ".join(evidence)
                evidence = process_evidence(evidence)

            if use_title and "page" in line:
                title = process_title(line["page"])

            if "label" in line:
                label = line["label"][0]
            else:
                label = self.get_dummy_label()

            text_a = claim
            text_b = None
            if not claim_only:
                text_b = f"{title} : {evidence}" if use_title else evidence

            examples.append(
                InputExample(
                    guid=guid,
                    text_a=text_a,
                    text_b=text_b,
                    label=label,
                )
            )
        return examples
