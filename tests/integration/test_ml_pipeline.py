import pytest

from src.train.features import features_generator
from src.train.pipeline import estimator, pipeline

# To quick mark the whole module:
pytestmark = pytest.mark.integration

# Remember that the fixtures defined in `conftests.py`
# are automatically discovered by pytest, hence are 
# available here without importing them.

@pytest.fixture(scope='module')
def fitted_features_generator(data):
    return features_generator.fit(*data)


@pytest.fixture(scope='module')
def features_set(fitted_features_generator, X):
    return fitted_features_generator.transform(X)


@pytest.fixture(scope='module')
def fitted_estimator(features_set, y):
    return estimator.fit(features_set, y)


@pytest.fixture(scope='module')
def fitted_pipeline(X, y):
    return pipeline.fit(X, y)

# Note: the following tests don't even use `assert` statements.
# Indeed, they are not really needed here: what we want to test 
# is simply that the different computations we do on the pipeline
# can be executed without error. 

def test_pipeline_can_be_fitted(X, y):
    pipeline.fit(X, y)

def test_pipeline_can_predict_after_fitting(fitted_pipeline, X):
    fitted_pipeline.predict(X)

def test_feature_generator_can_be_fitted(X, y):
    features_generator.fit(X, y)

def test_feature_generator_can_transform_after_fitting(
    fitted_features_generator, X):
    fitted_features_generator.transform(X)

def test_estimator_can_fit_on_feature_generation_output(features_set, y):
    estimator.fit(features_set, y)

def test_estimator_can_predict_on_feature_generation_output(features_set):
    estimator.predict(features_set)
