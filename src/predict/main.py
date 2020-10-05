from logging import getLogger
import time
from typing import List

import numpy as np

from .utils import load_model_from_cache, load_model_from_s3, parse
from ..utils import SklearnEstimator

logger = getLogger(__name__)


def predict(feed: List[dict]) -> np.ndarray:
    X = parse(feed)
    model = get_model()
    start = time.time()
    logger.debug('Running prediction...')

    prediction = model.predict(X)

    duration = time.time() - start
    logger.info(f'Prediction completed in {duration:.2f}s')

    return prediction


def get_model(from_cache: bool = True) -> SklearnEstimator:
    if from_cache:
        logger.debug('Trying to fetch model from cache...')
        try:
            model = load_model_from_cache()
        except FileNotFoundError:
            logger.info('Model not found in cache')
        else:
            logger.info('Model found in cache')
            return model

    logger.debug('Downloading model from S3 bucket...')
    return load_model_from_s3()
