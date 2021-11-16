from hashlib import sha1
from os.path import getmtime as get_modification_timestamp
from pathlib import Path
import pickle
import time
from typing import Union, Any

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from .config import CACHE_MAX_AGE, OUTPUT_DIR

SklearnEstimator = Union[Pipeline, BaseEstimator]  # used in type annotations


# NOTE
# ====
# This module uses Pickle to dump/load Python objects to/from files or memory.
#
# On very large Numpy arrays, using Joblib can be more efficient than Pickle,
# (especially for Python < 3.8) but, this is *not* a general rule. On simple
# data structures, Joblib can be much slower than Pickle.
#
# Ref: Olivier Grisel @ https://stackoverflow.com/a/12617603
#
# We use Pickle by default, as it enables extra features such as in-memory I/O
# (as opposed to file-only). An actual performance test could be  done to
# check if Joblib brings a signficant performance jump.
# ====


def write_binary_data_to_file(data: bytes, filepath: Union[Path, str]) -> None:
    directory = Path(filepath).parent
    if not directory.exists():
        directory.mkdir(parents=True)  # silently create all required folders

    with open(filepath, 'wb') as fp:
        fp.write(data)


def read_binary_data_from_file(filepath: Union[Path, str]) -> bytes:
    with open(filepath, 'rb') as fp:
        return fp.read()


def hasher(obj: Any) -> str:
    """Given any object, return an hexa string that uniquely identifies it."""
    # sha1 has a lower probability of collision and take approximately the
    # same amount of time to compute than md5
    return sha1(serialize(obj)).hexdigest()


def serialize(obj: Any) -> bytes:
    return pickle.dumps(obj)


def deserialize(data: bytes) -> Any:
    return pickle.loads(data)


def store_in_cache(obj: Any, filepath: Union[Path, str]) -> int:
    """Pickle any kind of Python object, and store in a file.

    The returned value is the number of bytes that were stored.
    """
    data = serialize(obj)
    write_binary_data_to_file(data, filepath)
    return len(data)


def load_from_cache(filepath: Union[Path, str]) -> Any:
    """Load a Python object pickled in a file located at `filepath`.

    This function will raise a `FileNotFoundError` if the cache file doesn't
    exist. It is the responsibility of the caller to deal with the error.
    """
    if cache_has_expired(filepath):
        remove_file(filepath)

    data = read_binary_data_from_file(filepath)
    return deserialize(data)


def remove_file(filepath: Union[Path, str]) -> None:
    filepath = Path(filepath)
    if not filepath.is_dir():
        filepath.unlink()


def cache_has_expired(filepath: Union[Path, str]) -> bool:
    """Return True if a cache file has expired, False if still valid."""
    try:
        last_modified = get_modification_timestamp(filepath)
    except FileNotFoundError:
        return False

    cache_age = time.time() - last_modified
    return cache_age > CACHE_MAX_AGE


def find_experiment_directory(model_id: str) -> Path:
    """Return the path of an experiment directory given a model's unique ID."""
    for file in OUTPUT_DIR.rglob('*'):
        # Look for an index file named after this model_id. Index files are
        # stored along with other experiments data by the ExperimentLog object.
        # Cf. acmemodel.train.log
        if model_id in file.name:
            return file.parent

    raise FileNotFoundError(f'Model #{model_id} not found in experiment logs.')
