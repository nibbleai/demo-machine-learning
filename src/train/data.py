"""
Module for I/O
"""
import logging
from pathlib import Path
from typing import Union

import pandas as pd

from ..config import DATA_DIR, ROOT_DIR

DATA_FILENAME = 'australian_open.csv'
DATA_FILEPATH = DATA_DIR / DATA_FILENAME

logger = logging.getLogger(__name__)


def load_raw_data(filepath: Union[Path, str] = DATA_FILEPATH) -> pd.DataFrame:
    filepath = Path(filepath)
    logger.debug(f"Loading raw data from '{filepath.relative_to(ROOT_DIR)}'")
    data = pd.read_csv(filepath)
    return data
