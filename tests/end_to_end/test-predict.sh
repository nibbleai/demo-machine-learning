#!/usr/bin/env bash

BASEDIR=$(git rev-parse --show-toplevel)

generate-test-data () {
  echo "Generating a sample of 10 data points..."
  python -m tests.generate_test_data 10
}

run-test () {
  echo "Running prediction..."
  python -m src.main --predict --disable-cache --input="$(< data/sample.json)"
  if [[ ! "$?" == 0 ]];then
    echo -e "\n\n---\nTest failed."
    exit 1
  else
    echo -e "\n\n---\nTest succeeded!"
  fi
}

# CDing to base directory is only required if
# the src module has not been installed via
# setup.py
cd $BASEDIR && generate-test-data && run-test