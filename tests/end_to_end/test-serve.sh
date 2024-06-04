#!/usr/bin/env bash

BASEDIR=$(git rev-parse --show-toplevel)
COMMAND="python -m src.main --serve"
BASEURL=http://localhost:5000


generate-test-data () {
  # Create a JSON file containing 10 observations to predict. The JSON is
  # formatted to match the dict format expected by the predict system
  echo "Generating a sample of 10 data points..."
  python -m tests.generate_test_data 10
}


launch-server () {
  echo "Starting webserver..."
  # Launch the flask server in a background process. We need to prevent the
  # process to be automatically hanged up by the shell
  nohup $COMMAND > /dev/null 2>&1 &
  sleep 2
}

run-ping () {
  status_code=$(curl -s -o /dev/null -w "%{http_code}" $BASEURL/ping)
  if [[ ! $status_code == 200 ]];then
    echo "Test failed on ping."
    exit 1
  fi
}

run-prediction () {
  status_code=$(curl -s -o /dev/null \
                    -w "%{http_code}" \
                    --header "Content-Type: application/json" \
                    --request POST \
                    --data "{\"data\": $(< data/sample.json)}" \
                    $BASEURL/predict)
  if [[ ! $status_code == 200 ]];then
    echo "Test failed on predict."
    exit 1
  fi
}


kill-server () {
  # Find the process in which the flask server is running and force-kill it
  # This method is quite hacky... :) Please, feel free to re-implement it using
  # a variable to share between launch and kill!
  ps aux | grep "$COMMAND" | grep -v grep | awk '{print $2}' | xargs kill -9
}

# CDing to base directory is only required if
# the src module has not been installed via
# setup.py
cd $BASEDIR && launch-server
run-ping
run-prediction
kill-server
echo "Success!"
