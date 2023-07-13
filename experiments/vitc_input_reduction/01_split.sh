#!/bin/bash

set -ex

# Get current dirname
expr="$(basename "$PWD")"

# Remove suffix until the first "_"
dataname="${expr%%_*}"

data_dir="../data/${dataname}"

# Since the training file is too big, split it into eight small files
split -l 17000 --numeric-suffixes "${data_dir}/train.jsonl" "${data_dir}/train.jsonl."

wc -l "${data_dir}/train.jsonl."*
