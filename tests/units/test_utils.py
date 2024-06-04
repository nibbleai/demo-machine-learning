from unittest.mock import patch, Mock
from random import random
from pathlib import Path
import pickle
import time

import pytest

from src.utils import (
    load_from_cache, hasher, remove_file, cache_has_expired, CACHE_MAX_AGE)


FILE_CONTENT = 'This is a temporary file'

# NOTE:
# In the following fixtures, we "override" pytest's builtin
# fixture `tmp_path` so we can manipulate file paths, 
# directory paths, and non-existing paths. `tmp_path`
# points to a directory, as written in the docs:
# https://docs.pytest.org/en/stable/tmpdir.html

@pytest.fixture
def tmp_filepath(tmp_path):
    filepath = tmp_path / 'cache_file.pkl'
    with open(filepath, 'wb') as f:
        f.write(pickle.dumps(FILE_CONTENT))
    yield filepath


@pytest.fixture
def tmp_path_non_existing(tmp_path):
    filepath = tmp_path / 'DOES_NOT_EXIST'
    yield filepath

# Patching the `serialize` function is not strictly necessary...
# You could also consider it as an implementation detail, and only focus on
# the actual output. Both approaches are valid.
@patch('src.utils.serialize')
def test_hasher(mock_serialize):
    """Check that hasher returns different strings for different objects."""
    # If we get 1000 different result, we consider the hasher to work correctly
    # You could set it to more or less, depending on your need.
    sample_size = 1000
    # If we decide to patch `serialize`, we don't want it to interfere with our test. 
    # We just need to be sure that it returns some bytes for the hasher. 
    # It's just a matter of context, and not mocking `serialize` is also acceptable.
    mock_serialize.side_effect = pickle.dumps

    hashes = [hasher(random()) for _ in range(sample_size)]

    assert len(set(hashes)) == len(hashes)
    assert all(isinstance(h, str) for h in hashes)


class TestRemoveFile:

    def test_non_existing(self, tmp_path_non_existing):
        with pytest.raises(FileNotFoundError):
            remove_file(tmp_path_non_existing)

    def test_existing(self, tmp_filepath):
        remove_file(tmp_filepath)
        assert not tmp_filepath.exists()

    @patch('src.utils.Path.unlink', autospec=True)
    def test_directory(self, mock_unlink, tmp_path):
        remove_file(tmp_path)
        mock_unlink.assert_not_called()


class TestCacheHasExpired:

    def test_non_existing(self, tmp_path_non_existing):
        assert not cache_has_expired(tmp_path_non_existing)

    def test_existing(self, tmp_filepath):
        # Technically speaking, the cache could have expired
        # if it was set to a very small value, which is not supposed to happend
        # in this project
        assert not cache_has_expired(tmp_filepath)

    @patch('src.utils.get_modification_timestamp', autospec=True)
    def test_existing_expired(self, mock_getmtime, tmp_filepath):
        mock_getmtime.return_value = time.time() - CACHE_MAX_AGE * 2
        assert cache_has_expired(tmp_filepath)
        mock_getmtime.assert_called_once_with(tmp_filepath)


class TestLoadFromCache:

    def test_non_existing(self, tmp_path_non_existing):
        with pytest.raises(FileNotFoundError):
            load_from_cache(tmp_path_non_existing)

    @patch('src.utils.cache_has_expired', autospec=True)
    def test_existing_non_expired(self, mock_has_expired, tmp_filepath):
        mock_has_expired.return_value = False

        res = load_from_cache(tmp_filepath)

        assert res == FILE_CONTENT

    @patch('src.utils.cache_has_expired', autospec=True)
    @patch('src.utils.deserialize', autospec=True)
    def test_existing_expired(
        self, mock_deserialize, mock_has_expired, tmp_filepath):
        mock_has_expired.return_value = True

        mock_deserialize.assert_not_called()
        with pytest.raises(FileNotFoundError):
            load_from_cache(tmp_filepath)
