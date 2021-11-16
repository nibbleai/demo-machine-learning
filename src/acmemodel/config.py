import logging
import logging.config
import os
from pathlib import Path

from dotenv import load_dotenv, find_dotenv

try:
    load_dotenv(find_dotenv(raise_error_if_not_found=True))
except OSError:
    raise RuntimeError("'.env' file not found.")

__LOC__ = Path(__file__).resolve()

ROOT_DIR = __LOC__.parents[1]
OUTPUT_DIR = ROOT_DIR / 'output'
DATA_DIR = ROOT_DIR / 'data'
LOG_DIR = ROOT_DIR / 'logs'

CACHE_DIR = ROOT_DIR / 'cache'
CACHE_MAX_AGE = 24 * 3600  # seconds

PROJECT_NAME = ROOT_DIR.name

MODEL_LOCAL_STORAGE_DIRECTORY = f'{OUTPUT_DIR}/models'

RANDOM_SEED = 42
TARGET = 'outcome'

AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
AWS_S3_BUCKET_NAME = os.environ.get('AWS_S3_BUCKET_NAME')

S3_KEY_PREFIX = os.environ['USERNAME']

# S3 storage keys for modelization data. We store the training logs and the
# training data along with the serialized model so we can compare with future
# experiments.
# NOTE: S3 takes care of versionning the actual object, so we can always
# rollback to an earlier version of the production model.
MODEL_S3_STORAGE_KEY = f'{S3_KEY_PREFIX}/production/model.pkl'
DATASET_S3_STORAGE_KEY = f'{S3_KEY_PREFIX}/production/training_dataset.pkl'
TRAINING_REPORT_S3_STORAGE_KEY = f'{S3_KEY_PREFIX}/production/report.json'

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'default': {
            'format': '[%(asctime)s]\t%(levelname)s\t%(message)s'
        }
    },
    'handlers': {
        'stream': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'default'
        },
        'file': {
            'level': 'INFO',
            'filename': f'{LOG_DIR}/app.log',
            'formatter': 'default',
            'class': 'logging.handlers.TimedRotatingFileHandler',
            'when': 'midnight',
            'backupCount': 7  # keep the logs from last week, delete the rest
        }
    },
    'loggers': {
        'src': {  # logging config for this project only
            'handlers': ['stream', 'file'],
            'level': 'DEBUG',
            'propagate': False
        },
        '': {  # all 3rd-party libraries
            'handlers': ['stream', 'file'],
            'level': 'WARNING'
        }
    }
}


def _create_directories():
    for directory in [OUTPUT_DIR, DATA_DIR, LOG_DIR, CACHE_DIR]:
        if not directory.exists():
            directory.mkdir()


def _configure_logging():
    logging.config.dictConfig(LOGGING_CONFIG)


def _setup():
    _create_directories()
    _configure_logging()


_setup()
