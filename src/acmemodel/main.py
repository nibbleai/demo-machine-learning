#!/usr/bin/env python3
import argparse
import json
from typing import Any

from . import utils


PARSER = argparse.ArgumentParser()
PARSER.add_argument('-f', '--features', action='store_true',  default=False,
                    help='Generate features')

PARSER.add_argument('-t', '--train', action='store_true', default=False,
                    help='Train the model with training data.')

PARSER.add_argument('-hp', '--hyperopt', action='store_true', default=False,
                    help='Train the model using hyperparameter optimization '
                         "(only valid with '--train')")

PARSER.add_argument('-p', '--predict', action='store_true', default=False,
                    help="Make a prediciton on sample data. "
                         "('--input' required)")

PARSER.add_argument('-s', '--serve', action='store_true', default=False,
                    help='Run a webserver locally to enable access to the '
                         'prediction service via HTTP requests.')

PARSER.add_argument('-d', '--deploy-model', action='store_true', default=False,
                    help='Deploy a serialized model to S3 for using in '
                         'production.')

PARSER.add_argument('--input',
                    help='A JSON-formatted string to use as feed for the '
                         'predict system.')

PARSER.add_argument('--disable-cache', action='store_true', default=False,
                    help='Disable the caching system used to improve '
                         'I/O performance.')


def main(args):
    # Import here to allow monkey-patching of `load_from_cache` at runtime
    from .deploy_model import main as run_deployment
    from .predict.main import predict as run_prediction_system
    from .serve.app import main as run_serving_system
    from .train.main import main as run_training_system
    from .train.features.main import main as run_features_generation

    if args.features:
        run_features_generation()
    elif args.train:
        run_training_system(args.hyperopt)
    elif args.deploy_model:
        run_deployment()
    elif args.serve:
        run_serving_system()
    elif args.predict:
        assert getattr(args, 'input') is not None, \
            "JSON-formatted data is required as '--input' parameter."
        feed = parse_input(args.input)
        run_prediction_system(feed)
    else:
        required = ['serve', 'train', 'predict', 'deploy-model']
        raise RuntimeError(
            f"One of the following flags is required: "
            f"{', '.join('--' + r for r in required)} "
        )


def disable_cache():
    """Monkey-patch the `load_from_cache` util function to always raise."""
    def cache_not_found(*args, **kwargs):
        raise FileNotFoundError
    utils.load_from_cache = cache_not_found


def parse_input(raw_input: str) -> Any:
    return json.loads(raw_input)


if __name__ == '__main__':
    args = PARSER.parse_args()
    if args.disable_cache:
        disable_cache()
    main(args)
