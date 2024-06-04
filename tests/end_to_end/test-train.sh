#!/usr/bin/env bash

BASEDIR=$(git rev-parse --show-toplevel)

test-train() {
    python -m src.main --train
    if [[ ! "$?" == 0 ]]; then
        echo -e "\n\n---\nTest failed."
        exit 1
    else
        echo -e "\n\n---\nTest succeeded."
    fi
}

test-train-w-hp() (
    python -m src.main --train -hp
    if [[ ! "$?" == 0 ]]; then
        echo -e "\n\n---\nTest failed."
        exit 1
    else
        echo -e "\n\n---\nTest succeeded."
    fi
)

# CDing to base directory is only required if
# the src module has not been installed via
# setup.py
cd $BASEDIR && test-train && test-train-w-hp