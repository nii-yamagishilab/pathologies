#!/bin/bash

set -ex

# Get current dirname
expr="$(basename "$PWD")"

# Remove suffix until the first "_"
dataname="${expr%%_*}"

python '../build_reduced_data.py' "${dataname}"

python '../build_random_data.py' "${dataname}"

