#!/bin/bash
#SBATCH --job-name=reduce_on_test
#SBATCH --out=reduce_on_test.%A_%a.log
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:tesla_a100:1
#SBATCH --array=1

conda_setup="/home/smg/$(whoami)/miniconda3/etc/profile.d/conda.sh"
if [[ -f "${conda_setup}" ]]; then
  #shellcheck disable=SC1090
  . "${conda_setup}"
  conda activate pathologies
fi

set -ex

pretrained=roberta-base
max_len=128

if [[ ! -f 'best_beta.txt' ]]; then
  echo "Cannot find best_beta.txt file"
  exit
fi
best_beta=$(cat "best_beta.txt")

# Get current dirname
expr="$(basename "$PWD")"

# Remove suffix until the first "_"
dataname="${expr%%_*}"

data_dir="../data/${dataname}"
prefix="${pretrained}-${max_len}-${best_beta}"
model_dir="${prefix}-mod"
out_dir="${prefix}-out"

for file in "${model_dir}/checkpoints"/*.ckpt; do
  [[ "${file}" -nt "${latest}" ]] && latest="${file}"
done

if [[ -z "${latest}" ]]; then
  echo "Cannot find any checkpoint in ${model_dir}"
  exit
fi

split='test'

out_file="${out_dir}/${split}.reduced.jsonl"
if [[ ! -f "${out_file}" ]]; then
  HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python '../../reduce.py' \
    --checkpoint_file "${latest}" \
    --in_file "${data_dir}/${split}.jsonl" \
    --out_file "${out_file}" \
    --batch_size 64 \
    --gpus 1
fi
