#!/bin/bash
#SBATCH --job-name=calib
#SBATCH --out=calib.%A_%a.log
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
model_dir="../${dataname}_cross_entropy/${pretrained}-${max_len}-mod"

for file in "${model_dir}/checkpoints"/*.ckpt; do
  [[ "${file}" -nt "${latest}" ]] && latest="${file}"
done

if [[ -z "${latest}" ]]; then
  echo "Cannot find any checkpoint in ${model_dir}"
  exit
fi

python '../../calibrate.py' \
  --checkpoint_file "${latest}" \
  --in_file "${data_dir}/dev.jsonl" \
  --gpus 1
