from collections.abc import Sequence
import logging
from typing import List

import pandas as pd

# It's OK to import static values from src
from src.config import DATA_DIR, TARGET

DATASET_LOCATION = DATA_DIR / 'australian_open.csv'


def get_test_datapoints(count: int = 1) -> List[dict]:
    """Return `count` observations taken randomly from the raw data.

    The returned value is a list of datapoint dictionnaries, ready to be
    transformed to a JSON object if necessary.
    """
    df = get_raw_dataframe()
    X = df.drop([TARGET], axis=1)

    return X.sample(count).to_dict(orient='records')


def get_raw_dataframe() -> pd.DataFrame:
    return pd.read_csv(DATASET_LOCATION)


def disable_src_logs() -> None:
    for name, logger in logging.Logger.manager.loggerDict.items():
        if isinstance(logger, logging.PlaceHolder):
            continue
        if 'src' in name:
            logger.addHandler(logging.NullHandler())
            logger.propagate = False


def create_dataset_from_dictionary(data_as_dict):

    def force_list(value):
        # Dict values *must* be a list of values so it can be converted to a
        # Dataframe. Convert scalar values to a single-list item.
        if isinstance(value, Sequence):
            return list(value)
        return [value]

    dict_formatted = {k: force_list(v) for k, v in data_as_dict.items()}
    return pd.DataFrame(dict_formatted)
