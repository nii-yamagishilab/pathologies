#!/bin/bash

set -ex

# Download datasets and uncompress
url='https://github.com/TalSchuster/talschuster.github.io/raw/master/static'
for name in \
    'vitaminc' \
    'vitaminc_real' \
    'vitaminc_synthetic'; do
  if [[ ! -f "${name}.zip" ]]; then
    wget "${url}/${name}.zip"
  fi
  if [[ ! -d "${name}" ]]; then
    unzip "${name}.zip"
  fi
done

url="${url}/vitaminc_baselines"
name='fever'
if [[ ! -f "${name}.zip" ]]; then
  wget "${url}/${name}.zip"
fi
if [[ ! -d "${name}" ]]; then
  unzip "${name}.zip"
fi

# Extract train/dev sets for vitaminc_{real,synthetic}
for split in 'train' 'dev'; do
  real="vitaminc_real/${split}.jsonl"
  if [[ ! -f "${real}" ]]; then
    grep ': "real"' "vitaminc/${split}.jsonl" > "${real}"
    wc -l "${real}"
  fi
  syn="vitaminc_synthetic/${split}.jsonl"
  if [[ ! -f "${syn}" ]]; then
    grep -v ': "real"' "vitaminc/${split}.jsonl" > "${syn}"
    wc -l "${syn}"
  fi
done

ln -nfs vitaminc_synthetic vitc
