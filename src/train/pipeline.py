"""Pipeline
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion

from .features import features_generator

SEED = 42

RF_PARAMS = {
    'n_estimators': 10,
    'random_state': SEED
}

PARAM_GRID = {
    'estimator__max_depth': [10, 20],
}

estimator = RandomForestClassifier(**RF_PARAMS)

pipeline = Pipeline([
    ('features', features_generator),
    ('estimator', estimator)
])
