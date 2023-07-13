#!/bin/bash
#SBATCH --job-name=train
#SBATCH --out=train.%A_%a.log
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

seed=3435
pretrained=roberta-base
lr=3e-5
max_len=128
batch_size=16
accumulate=1

# Get current dirname
expr="$(basename "$PWD")"

# Remove suffix until the first "_"
dataname="${expr%%_*}"

# Remove prefix until "_"
loss="${expr#*_}"

data_dir="../data/${dataname}"
model_dir="${pretrained}-${max_len}-mod"

if [[ -d "${model_dir}" ]]; then
  echo "${model_dir} exists! Skip training."
  exit
fi

python '../../train.py' \
  --data_dir "${data_dir}" \
  --default_root_dir "${model_dir}" \
  --pretrained_model_name "${pretrained}" \
  --max_seq_length "${max_len}" \
  --seed "${seed}" \
  --cache_dir "/local/$(whoami)" \
  --overwrite_cache \
  --loss_function "${loss}" \
  --train_on_reduced_inputs \
  --max_epochs 10 \
  --learning_rate "${lr}" \
  --train_batch_size "${batch_size}" \
  --accumulate_grad_batches "${accumulate}" \
  --adafactor \
  --warmup_ratio 0.02 \
  --gradient_clip_val 1.0 \
  --precision 16 \
  --deterministic true \
  --gpus 1
