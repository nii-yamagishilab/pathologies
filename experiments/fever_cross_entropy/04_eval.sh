#!/bin/bash

set -ex

pretrained=roberta-base
max_len=128

# Get current dirname
expr="$(basename "$PWD")"

# Remove suffix until the first "_"
dataname="${expr%%_*}"

data_dir="../data/${dataname}"
prefix="${pretrained}-${max_len}"
out_dir="${prefix}-out"

for split in 'dev' 'test'; do
  eval_file="${out_dir}/eval.${split}.txt"
  python '../../evaluate.py' \
    --gold_file "${data_dir}/${split}.jsonl" \
    --prob_file "${out_dir}/${split}.prob" \
    --reduced_file "${out_dir}/${split}.reduced.jsonl" \
    --out_file "${eval_file}" \
    --plot_diagram
done
