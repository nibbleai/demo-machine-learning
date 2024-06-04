import pandas as pd
import pytest


EXPECTED_COLUMNS_AND_DTYPES = [
    ('rally', 'int64'),
    ('serve', 'int64'),
    ('hitpoint', 'object'),
    ('speed', 'float64'),
    ('net.clearance', 'float64'),
    ('distance.from.sideline', 'float64'),
    ('depth', 'float64'),
    ('outside.sideline', 'bool'),
    ('outside.baseline', 'bool'),
    ('player.distance.travelled', 'float64'),
    ('player.impact.depth', 'float64'),
    ('player.impact.distance.from.center', 'float64'),
    ('player.depth', 'float64'),
    ('player.distance.from.center', 'float64'),
    ('previous.speed', 'float64'),
    ('previous.net.clearance', 'float64'),
    ('previous.distance.from.sideline', 'float64'),
    ('previous.depth', 'float64'),
    ('opponent.depth', 'float64'),
    ('opponent.distance.from.center', 'float64'),
    ('same.side', 'bool'),
    ('previous.hitpoint', 'object'),
    ('previous.time.to.net', 'float64'),
    ('server.is.impact.player', 'bool'),
    ('id', 'int64'),
    ('outcome', 'object')
]

EXPECTED_COLUMNS = [c for c, d in EXPECTED_COLUMNS_AND_DTYPES]


@pytest.mark.integration
class TestLoadRawData:

    def test_it_should_return_a_dataframe(self, raw_data):
        assert isinstance(raw_data, pd.DataFrame)

    def test_it_has_expected_number_of_columns(self, raw_data):
        assert len(raw_data.columns) == len(EXPECTED_COLUMNS)

    def test_columns_match_expectation(self, raw_data):
        assert set(raw_data.columns) == set(EXPECTED_COLUMNS)

    @pytest.mark.parametrize('col_name', EXPECTED_COLUMNS)
    def test_it_has_column(self, raw_data, col_name):
        assert col_name in raw_data.columns

    @pytest.mark.parametrize('col_name, expected', EXPECTED_COLUMNS_AND_DTYPES)
    def test_dtypes(self, col_name, expected, raw_data):
        assert raw_data[col_name].dtype == expected
