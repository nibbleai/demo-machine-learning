from logging import getLogger
from pathlib import Path
from typing import Union

from .pipeline import pipeline, PARAM_GRID
from ..utils import write_binary_data_to_file, serialize, SklearnEstimator

logger = getLogger(__name__)


def get_model() -> SklearnEstimator:
    return pipeline


def get_param_grid() -> dict:
    return PARAM_GRID


def save_model(model: SklearnEstimator, filepath: Union[Path, str]) -> None:
    data = serialize(model)
    write_binary_data_to_file(data, filepath)
