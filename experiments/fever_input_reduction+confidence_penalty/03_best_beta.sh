#!/bin/bash

# Get current dirname
expr="$(basename "$PWD")"

# Remove suffix until the first "_"
dataname="${expr%%_*}"

# Remove prefix until "_"
loss="${expr#*_}"

python '../find_best_beta.py' "${dataname}" "${loss}"
