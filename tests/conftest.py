"""
Any fixture implemented in conftest.py is made automatically available by
pytest to every functions of the test suite.
"""
import pytest

from src.train.dataset import get_testing_data
from src.train.data import load_raw_data


@pytest.fixture(scope='session')
def raw_data():
    return load_raw_data()


@pytest.fixture(scope='session')
def small_data(raw_data):
    return raw_data[:300]


@pytest.fixture(scope='session')
def data(small_data):
    X_test, y_test = get_testing_data(small_data)
    return X_test, y_test


@pytest.fixture(scope='session')
def X(data):
    X, _ = data
    return X


@pytest.fixture(scope='session')
def y(data):
    _, y = data
    return y
