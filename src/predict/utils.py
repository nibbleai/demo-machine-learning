from collections import defaultdict
from functools import reduce
from typing import List

from pandas import DataFrame

from ..aws import download_data_from_s3
from ..config import CACHE_DIR, MODEL_S3_STORAGE_KEY
from ..utils import (load_from_cache, write_binary_data_to_file, deserialize,
                     SklearnEstimator)


MODEL_CACHE_DIRECTORY = CACHE_DIR / 'models'
MODEL_CACHE_KEY = MODEL_CACHE_DIRECTORY / 'model.pkl'


def load_model_from_cache() -> SklearnEstimator:
    return load_from_cache(MODEL_CACHE_KEY)


def load_model_from_s3() -> SklearnEstimator:
    model_data = download_data_from_s3(MODEL_S3_STORAGE_KEY)
    # We don't use the `store_in_cache` util function because we already have
    # a serialized version of the model, but the outcome is the same.
    write_binary_data_to_file(model_data, MODEL_CACHE_KEY)
    return deserialize(model_data)


def parse(feed: List[dict]) -> DataFrame:
    """Convert a list of individual data points to a pandas DataFrame."""
    return DataFrame(feed)
