"""
Dataset manager
"""
import logging
from typing import Tuple, Sequence

from sklearn.model_selection import train_test_split as _train_test_split
from pandas import DataFrame

from .. import config
from ..utils import write_binary_data_to_file, serialize

CACHE_DIRECTORY = config.CACHE_DIR / 'datasets'

logger = logging.getLogger(__name__)


def get_training_data(dataset: DataFrame) -> Tuple[DataFrame, DataFrame]:
    X, y = split_labels(dataset)
    X_train, _, y_train, _ = train_test_split(X, y)
    return X_train, y_train


def get_testing_data(dataset: DataFrame) -> Tuple[DataFrame, DataFrame]:
    X, y = split_labels(dataset)
    _, X_test, _, y_test = train_test_split(X, y)
    return X_test, y_test


def train_test_split(X: Sequence, y: Sequence) -> tuple:
    """Custom wrapper around Sklearn's `train_test_split` function.

    By using this wrapper in our code, we make sure that we always use the
    split parameters.
    """
    return _train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_SEED, stratify=y)


def split_labels(df: DataFrame,
                 target: str = config.TARGET) -> Tuple[DataFrame, DataFrame]:
    y = df[target]
    X = df.drop([target], axis=1)
    return X, y


def save_dataset(dataset: DataFrame, filepath):
    return write_binary_data_to_file(serialize(dataset), filepath)
