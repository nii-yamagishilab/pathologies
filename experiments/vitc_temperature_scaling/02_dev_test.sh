#!/bin/bash
#SBATCH --job-name=dev_test
#SBATCH --out=dev_test.%A_%a.log
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

# Get current dirname
expr="$(basename "$PWD")"

# Remove suffix until the first "_"
dataname="${expr%%_*}"

data_dir="../data/${dataname}"
prefix="${pretrained}-${max_len}"
model_dir="../${dataname}_cross_entropy/${prefix}-mod"
out_dir="${prefix}-out"

for file in "${model_dir}/checkpoints"/*.ckpt; do
  [[ "${file}" -nt "${latest}" ]] && latest="${file}"
done

if [[ -z "${latest}" ]]; then
  echo "Cannot find any checkpoint in ${model_dir}"
  exit
fi

mkdir -p "${out_dir}"

for split in 'dev' 'test'; do
  out_file="${out_dir}/${split}.prob"
  if [[ ! -f "${out_file}" ]]; then
    HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python '../../predict.py' \
      --checkpoint_file "${latest}" \
      --in_file "${data_dir}/${split}.jsonl" \
      --out_file "${out_file}" \
      --temperature "$(cat 'best_temperature.txt')" \
      --batch_size 128 \
      --gpus 1
  fi
done
