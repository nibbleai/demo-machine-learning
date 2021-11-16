"""
Module for creating KFold
"""
import logging

from sklearn.model_selection import StratifiedKFold

from src import config


logger = logging.getLogger(__name__)


def gen_kfold(n_folds=5):
    """ Generate kfold """
    return StratifiedKFold(n_splits=n_folds, shuffle=True,
                           random_state=config.RANDOM_SEED)
