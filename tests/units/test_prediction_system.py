import random
from unittest.mock import patch, Mock

import pandas as pd
import pytest

from src.predict.main import predict, get_model, parse
from tests.utils import get_test_datapoints


@patch('src.predict.main.get_model')
@patch('src.predict.main.parse')
def test_predict_function(mock_parse, mock_get_model):
    """Check that the predict wrapper is calling the model.predict() method."""
    # NOTE: this "long" Arrange part suggests that there is room for
    # improvements in the implementation of the `predict` function
    fake_feed = Mock()
    fake_prediction = Mock()
    fake_model = Mock()
    fake_input = get_test_datapoints()
    fake_model.predict.return_value = fake_prediction
    mock_parse.return_value = fake_input
    mock_get_model.return_value = fake_model

    prediction = predict(fake_feed)

    assert prediction == fake_prediction
    fake_model.predict.assert_called_once_with(fake_input)


@pytest.mark.parametrize('use_cache', [True, False])
@pytest.mark.parametrize('cache_exists', [True, False])
@patch('src.predict.main.load_model_from_s3')
@patch('src.predict.main.load_model_from_cache')
def test_get_model(mock_from_cache, mock_from_s3, use_cache, cache_exists):
    """Check that model can be fetched from S3 or from cache.

    Even if use_cache is True, if the cache file does not exist, we must ensure
    that the function falls back on downloading from S3.
    """
    fake_model = Mock()
    mock_from_s3.return_value = fake_model
    if cache_exists:
        mock_from_cache.return_value = fake_model
    else:
        mock_from_cache.side_effect = FileNotFoundError

    model = get_model(use_cache)

    assert model == fake_model
    if use_cache and cache_exists:
        mock_from_cache.assert_called_once()
    elif use_cache and not cache_exists:
        mock_from_cache.assert_called_once()  # raised a FileNotFound error
        mock_from_s3.assert_called_once()
    else:
        mock_from_s3.assert_called_once()


@pytest.mark.parametrize('observation_count', [1, 10, 100, 1000])
@pytest.mark.parametrize('key_count', [5, 10, 20, 50])
def test_feed_parser(observation_count, key_count):
    """Check that the feed parser converts list of dict to DataFrames."""
    fake_keys = [f'key_{i}' for i in range(key_count)]
    fake_observations = [
        {k: random.random() for k in fake_keys}
        for _ in range(observation_count)
    ]

    converted = parse(fake_observations)

    assert isinstance(converted, pd.DataFrame)
    assert len(converted) == observation_count
    assert set(converted.columns) == set(fake_keys)
