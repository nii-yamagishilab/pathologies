#!/bin/bash

set -ex

# Get current dirname
expr="$(basename "$PWD")"

# Remove suffix until the first "_"
dataname="${expr%%_*}"

data_dir="../data/${dataname}"

# Merge small files into one big file
cat "${data_dir}/train.jsonl."*.reduced > "${data_dir}/train.reduced.jsonl"

tail -n 3 "${data_dir}/train.jsonl" "${data_dir}/train.reduced.jsonl"

wc -l "${data_dir}/train.jsonl" "${data_dir}/train.reduced.jsonl"
