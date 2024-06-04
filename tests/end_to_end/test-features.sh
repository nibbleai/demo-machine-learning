#!/usr/bin/env bash


BASE_DIR=$(git rev-parse --show-toplevel)

# CDing to base directory is only required if
# the src module has not been installed via
# setup.py
cd $BASE_DIR && python -m src.main --features
