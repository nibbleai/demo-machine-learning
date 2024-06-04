"""
This is not so much a "test" than a mini-application in itself.

Model validation is hard to automate, so we still rely on human intervention
here. This is a light introduction step to model performance comparison.
"""
from functools import partial
import json
from pathlib import Path
from statistics import stdev, mean
from typing import Tuple

from pandas import DataFrame

# It's OK to import from src here, because we are not trying to test the
# application itself: we only want to test the performance of an output
# of the application, namely, a fitted model.
from src.aws import download_data_from_s3
from src.config import (CACHE_DIR as ROOT_CACHE, DATASET_S3_STORAGE_KEY,
                        TRAINING_REPORT_S3_STORAGE_KEY)
from src.deploy_model import MODEL_FILENAME
from src.train.dataset import split_labels, train_test_split
from src.train.train import cross_validate, evaluate
from src.utils import (read_binary_data_from_file, write_binary_data_to_file,
                       deserialize, find_experiment_directory, SklearnEstimator)


PROD_BUCKET_NAME = 'australian-open-outcome-predictions'

PROD_REPORT_CACHE_KEY = 'production_report.json'
EVALUATION_DATA_CACHE_KEY = 'evaluation_data.pkl'

CACHE_DIR = ROOT_CACHE / 'tests'

PROD_REPORT_CACHE_PATH = CACHE_DIR / PROD_REPORT_CACHE_KEY
EVALUATION_DATA_CACHE_PATH = CACHE_DIR / EVALUATION_DATA_CACHE_KEY

CACHE_KEY_TO_S3_KEY = {
    PROD_REPORT_CACHE_KEY: TRAINING_REPORT_S3_STORAGE_KEY,
    EVALUATION_DATA_CACHE_KEY: DATASET_S3_STORAGE_KEY
}

# ----------------------------------------------------------------------------
# Utility functions to fetch objects from production bucket
#

def get_from_cache_or_download_from_s3(cache_path: Path):
    cache_key = cache_path.name
    s3_key = CACHE_KEY_TO_S3_KEY[cache_key]

    if cache_path.exists():
        data = read_binary_data_from_file(cache_path)
    else:
        data = download_data_from_s3(s3_key, PROD_BUCKET_NAME)

    write_binary_data_to_file(data, cache_path)  # store in cache
    return deserialize(data) if cache_key.endswith('.pkl') else data.decode()


# Get raw data that were used to fit the production model
get_evaluation_data = partial(get_from_cache_or_download_from_s3,
                              EVALUATION_DATA_CACHE_PATH)
# Get JSON report from production model training
get_production_report = partial(get_from_cache_or_download_from_s3,
                                PROD_REPORT_CACHE_PATH)


def get_production_metrics() -> Tuple[float, float, float]:
    production_report = json.loads(get_production_report())

    log_loss = production_report['test_log_loss']
    cv_mean = production_report['mean_cv_log_loss']
    cv_std = production_report['std_cv_log_loss']

    return log_loss, cv_mean, cv_std

# ----------------------------------------------------------------------------


def main() -> None:
    """Run an evaluation metrics benchmark of a given model VS production."""
    evaluation_data = get_evaluation_data()

    model_id = input('Enter the ID of the model you want to benchmark:\n>  ')
    model = get_model_from_id(model_id)
    log_loss, mean_cv, std_cv = get_metrics(model, evaluation_data)

    display_benchmark(model_id, log_loss, mean_cv, std_cv)


def get_model_from_id(model_id: str) -> SklearnEstimator:
    directory = find_experiment_directory(model_id)
    model_data = read_binary_data_from_file(directory / MODEL_FILENAME)
    return deserialize(model_data)


def get_metrics(model: SklearnEstimator,
                dataset: DataFrame) -> Tuple[float, float, float]:
    X, y = split_labels(dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    log_loss = evaluate(model, X_test, y_test)
    cv_log_losses = cross_validate(model, X_train, y_train)

    return log_loss, mean(cv_log_losses), stdev(cv_log_losses)


def display_benchmark(model_id: str, log_loss: float,
                      mean_cv_log_loss: float, std_cv_log_loss: float) -> None:
    """Print a benchmark to compare evaluation metrics with prod model."""
    template = ' {0:<17}|{1:^10.4f}|{2:^10.4f}|{3:^10.4f}'
    header = ' {0:<17}|{1:^10}|{2:^10}|{3:^10}'.format(
        'model id', 'log loss', 'cv mean', 'cv std')

    this_summary = template.format(
        f'{model_id[:10]}...',
        log_loss,
        mean_cv_log_loss,
        std_cv_log_loss
    )

    prod_log_loss, prod_cv_mean, prod_cv_std = get_production_metrics()
    prod_summary = template.format(
        'production',
        prod_log_loss,
        prod_cv_mean,
        prod_cv_std
    )

    print('\n')
    print('•' * 50)
    print('•{:^48}•'.format('Benchmark results'))
    print('•' * 50)
    print('\n')
    print(header)
    print('-' * 50)
    print(this_summary)
    print(prod_summary)
    print('-' * 50)
    print('\nThese results were calculated with the same set of data.\n')


if __name__ == '__main__':
    main()
