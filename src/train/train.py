from logging import getLogger
from typing import Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import log_loss

from .dataset import train_test_split, split_labels
from .fold import gen_kfold
from .log import TrainingLogger
from .model import get_param_grid
from ..utils import SklearnEstimator

logger = getLogger(__name__)


SCORING = 'neg_log_loss'


def train(estimator: SklearnEstimator,
          data: DataFrame,
          optimize: bool = False,
          log: bool = True) -> Tuple[float, np.ndarray]:
    logger.debug(f'Training {estimator.__class__.__name__} on dataset...')

    X, y = split_labels(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    if optimize:
        best_estimator, cv_log_losses = hyperopt(
            estimator, X_train, y_train, param_grid=get_param_grid())
    else:
        cv_log_losses = cross_validate(estimator, X_train, y_train)
        best_estimator = estimator

    best_estimator.fit(X_train, y_train)
    test_log_loss = evaluate(best_estimator, X_test, y_test)

    logger.info(f'Mean log loss on CV: {cv_log_losses.mean():.4f}')
    logger.info(f'Std log loss on CV: {np.std(cv_log_losses):.4f}')
    logger.info(f'Log loss on test set: {test_log_loss:.4f}')

    if log:
        log_experiment(estimator, data, test_log_loss, cv_log_losses)

    return test_log_loss, cv_log_losses


def cross_validate(estimator: SklearnEstimator,
                   X: DataFrame,
                   y: DataFrame) -> np.ndarray:
    return -cross_val_score(estimator, X, y,
                            scoring=SCORING, cv=gen_kfold())


def hyperopt(estimator: SklearnEstimator,
             X: DataFrame,
             y: DataFrame,
             param_grid: dict) -> Tuple[SklearnEstimator, np.ndarray]:
    logger.info(
        'Performing hyperparameter optimization with GridSearchCV...')

    gs = GridSearchCV(estimator, param_grid, scoring=SCORING, cv=gen_kfold())
    gs.fit(X, y)

    best_loss = -gs.best_score_
    logger.info(f'Best loss: {best_loss:.4f}')

    cv_results = pd.DataFrame(gs.cv_results_).sort_values(by='rank_test_score')

    regex_log_loss_cv_cols = '^split[0-9]+_test_score'
    best_cv_losses = -cv_results.loc[gs.best_index_] \
        .filter(regex=regex_log_loss_cv_cols)

    return gs.best_estimator_, best_cv_losses


def evaluate(estimator: SklearnEstimator, X: DataFrame, y: DataFrame) -> float:
    logger.debug('Evaluating performance of model...')
    y_prob = estimator.predict_proba(X)
    return log_loss(y, y_prob)


def log_experiment(estimator: SklearnEstimator,
                   dataset: DataFrame,
                   test_log_loss: float,
                   cv_log_losses: np.ndarray) -> None:
    """Record experiment data into a dedicated directory."""
    log = TrainingLogger(estimator, dataset, test_log_loss, cv_log_losses)
    log.save()
