"""Util script to generate testing data.

Example:
    Generate a JSON file containing 20 data points

    $ python -m tests.generate_test_data 20

If no parameter is given at runtine, default to 10.
"""
import json
import sys

from src.config import DATA_DIR
from .utils import get_test_datapoints

SAMPLE_FILE = DATA_DIR / 'sample.json'


def main(count) -> None:
    data = get_test_datapoints(count)
    with open(SAMPLE_FILE, 'w') as fp:
        fp.write(json.dumps(data, indent=2))


if __name__ == '__main__':
    try:
        count = int(sys.argv[1])
    except IndexError:
        count = 10
    main(count)
