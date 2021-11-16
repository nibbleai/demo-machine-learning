from datetime import datetime
import json
from logging import getLogger
from pathlib import Path

import numpy as np
from pandas import DataFrame

from .dataset import save_dataset
from .encoder import JSONEncoder as CustomJSONEncoder
from .model import save_model
from ..config import OUTPUT_DIR, ROOT_DIR
from ..utils import SklearnEstimator, hasher

logger = getLogger(__name__)

MODEL_FILENAME = 'model.pkl'
DATASET_FILENAME = 'dataset.pkl'
REPORT_FILENAME = 'report.json'


class TrainingLogger:
    """An simple object to keep track of trainings.

    Public interface
    ================
    Attributes:
        * `report` (dict) - Summary of experiment parameters
        * `to_json` (str) - JSON reprensation of `report`
        * `storage_directory` (Path) - Directory where the artifacts are stored
        * `id` (str) - Unique identifier based on the fitted model state

    Method:
        * `save()` - Store the experiment artifacts in `storage_directory`
    """
    def __init__(self,
                 estimator: SklearnEstimator,
                 dataset: DataFrame,
                 test_log_loss: float,
                 cv_losses: np.ndarray):
        self._estimator = estimator
        self._dataset = dataset
        self._test_log_loss = test_log_loss
        self._cv_log_losses = cv_losses

        self.timestamp = datetime.now().strftime('%Y%m%d-%Hh%Mm%Ss')
        self._report = None

    @property
    def report(self) -> dict:
        if self._report is None:
            self._report = {
                'test_log_loss': self._test_log_loss,
                'mean_cv_log_loss': np.mean(self._cv_log_losses),
                'std_cv_log_loss': np.std(self._cv_log_losses),
                'cv_log_losses': list(self._cv_log_losses),
                'estimator': {
                    'id': hasher(self._estimator),
                    'type': self._estimator.__class__.__name__,
                    'params': self._estimator.get_params()
                }
            }
        return self._report

    @property
    def to_json(self):
        return json.dumps(self.report, indent=2, cls=CustomJSONEncoder)

    @property
    def storage_directory(self) -> Path:
        return OUTPUT_DIR / self.timestamp

    def save(self) -> None:
        """Save the experiment artifacts under `storage_directory`."""
        relative_storage_dir = self.storage_directory.relative_to(ROOT_DIR)
        logger.info(
            f"Storing experiment artifacts under '{relative_storage_dir}'"
        )
        save_model(self._estimator, f'{self.storage_directory}/{MODEL_FILENAME}')
        save_dataset(self._dataset, f'{self.storage_directory}/{DATASET_FILENAME}')
        self._save_report()
        self._create_index_file()

    def _save_report(self) -> None:
        # Store the whole report
        with open(f'{self.storage_directory}/{REPORT_FILENAME}', 'w') as fp:
            fp.write(self.to_json)

    def _create_index_file(self):
        # Create an empty file named after the model ID to easily look for it
        index = self.report['estimator']['id']
        file = self.storage_directory / f'.{index}'
        file.touch()
