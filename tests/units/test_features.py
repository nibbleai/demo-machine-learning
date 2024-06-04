import numpy as np
import pytest

from src.train.features import FEATURES_LIST
from src.train.features.features import (
    Speed,
    NetClearance,
    DistanceFromSideline,
    Depth,
    PlayerDistanceTravelled,
    PlayerImpactDepth,
    PreviousDistanceFromSideline,
    PreviousTimeToNet,
    Hitpoint,
    Out,
    WeirdNetClearance,
    DistanceTravelledRatio
)

from tests.utils import create_dataset_from_dictionary


@pytest.mark.parametrize('Feature', FEATURES_LIST)
class TestFeaturesBase:

    def test_attributes(self, Feature):
        assert hasattr(Feature, 'fit')
        assert hasattr(Feature, 'transform')
        assert hasattr(Feature, 'fit_transform')


@pytest.mark.parametrize('Feature', FEATURES_LIST)
class TestFeaturesUsage:

    def test_can_be_fitted(self, Feature, X, y):
        Feature().fit(X, y)

    def test_can_be_transformed_after_fitting(self, Feature, X, y):
        feat = Feature()
        feat.fit(X, y)
        X_tr = feat.transform(X)
        assert X_tr is not None

    def test_shape(self, Feature, X, y):
        X_tr = Feature().fit_transform(X, y)
        assert len(X_tr.shape) == 2, 'Need shape of 2'

    def test_preserve_shape(self, Feature, X, y):
        X_tr = Feature().fit_transform(X, y)

        assert X_tr.shape[0] == X.shape[0]


@pytest.mark.parametrize('column_name, Feature', [
    ('speed', Speed),
    ('net.clearance', NetClearance),
    ('distance.from.sideline', DistanceFromSideline),
    ('depth', Depth),
    ('player.distance.travelled', PlayerDistanceTravelled),
    ('player.impact.depth', PlayerImpactDepth),
    ('previous.distance.from.sideline', PreviousDistanceFromSideline),
    ('previous.time.to.net', PreviousTimeToNet),
])
class TestColumnExtractors:

    def test_it_returns_only_one_column(self, column_name, Feature, X, y):
        X_tr = Feature().fit_transform(X, y)
        assert X_tr.shape[1] == 1

    def test_identity(self, column_name, Feature, X, y):
        X_tr = Feature().fit_transform(X, y)
        assert np.array_equal(X_tr.squeeze(), X[column_name])

    def test_fails_if_column_is_missing(self, column_name, Feature, X, y):
        X_new = X.drop(column_name, axis=1)

        feature = Feature().fit(X, y)

        with pytest.raises(KeyError):
            feature.transform(X_new)


class TestHitpoint:

    def test_columns_count_on_fit_transform(self):
        data_as_dict = {
            'hitpoint': ['F', 'B', 'V', 'U', 'F']
        }
        data = create_dataset_from_dictionary(data_as_dict)

        res = Hitpoint().fit_transform(data)

        assert res.shape[1] == 3

    @pytest.mark.parametrize('values', [
        ['F'],
        ['F', 'B'],
        ['F', 'B', 'V'],
        ['F', 'B', 'V', 'U'],
        ['F', 'B', 'V', 'U', 'F']
    ])
    def test_columns_count_when_transform_with_different_uniques(self, values):
        data = create_dataset_from_dictionary({
            'hitpoint': ['F', 'B', 'V', 'U', 'F']
        })
        new_data = create_dataset_from_dictionary({
            'hitpoint': values
        })
        feat = Hitpoint().fit(data)
        res = feat.transform(new_data)

        assert res.shape[1] == 3


class TestOut:

    @pytest.mark.parametrize('outside_baseline, outside_sideline, expected', [
        (True, True, True),
        (True, False, True),
        (False, True, True),
        (False, False, False)
    ])
    def test_values(self, outside_baseline, outside_sideline, expected):
        data_as_dict = {
            'outside.sideline': outside_baseline,
            'outside.baseline': outside_sideline
        }
        data = create_dataset_from_dictionary(data_as_dict)

        res = Out().fit_transform(data)

        assert res == expected

    @pytest.mark.parametrize('column_name', [
        'outside.baseline',
        'outside.sideline',
        'random_column_name'
    ])
    def test_fails_if_columns_missing(self, column_name):
        data_as_dict = {
            column_name: True
        }
        data = create_dataset_from_dictionary(data_as_dict)

        with pytest.raises(KeyError):
            Out().fit_transform(data)


@pytest.mark.parametrize('net_clearance, expected', [
    # We expect only value between -0.946 and -0.947 to be
    # flagged as weird net clearances
    (0.947, False),
    (0.5, False),
    (0, False),
    (-0.5, False),
    (-0.946, False),
    (-0.947, True),
    (-0.948, False),
    (-0.99, False)
])
def test_weird_net_clearance(net_clearance, expected):
    X = create_dataset_from_dictionary({
        'net.clearance': net_clearance
    })

    res = WeirdNetClearance().fit_transform(X)

    assert res[0][0] == expected


class TestDistanceTravelledRatio:

    @pytest.mark.parametrize('x, y', [
        (0, 0),
        (0, 5),
        (3, 0),
        (4, 4)
    ])
    def test_when_flying_distance_is_null(self, x, y):
        # Point of departure and point of arrival are the same.
        # We are checking that even in this edge case, we dont
        # end up with a NAN value, nor with an error of any kind.
        data_as_dict = {
            'player.distance.from.center': x,
            'player.depth': y,
            'player.impact.distance.from.center': x,
            'player.impact.depth': y,
            'player.distance.travelled': 0
        }
        data = create_dataset_from_dictionary(data_as_dict)

        res = DistanceTravelledRatio().fit_transform(data)
        assert not np.isnan(res)
        # fit_transform returns a single item numpy array: 
        # we need to unpack 
        assert res[0] == 1

    @pytest.mark.parametrize('x1, y1, x2, y2, distance, expected', [
        (1, 0, 0, 0, 1, 1),
        (1, 0, 0, 0, 2, 2),
        (0, 1, 0, 0, 1, 1),
        (0, 1, 0, 0, 2, 2),
        (0, 0, 1, 0, 1, 1),
        (0, 0, 1, 0, 2, 2),
        (0, 0, 0, 1, 1, 1),
        (0, 0, 0, 1, 2, 2),
        (1, 1, 1, 1, 0, 1),
        (0, 0, 3, 4, 5, 1),
        (3, 4, 0, 0, 5, 1),
        (4, 6, 1, 2, 10, 2)
    ])
    def test_flying_distance_not_null(self, x1, y1, x2, y2, distance, expected):
        data_as_dict = {
            'player.distance.from.center': x1,
            'player.depth': y1,
            'player.impact.distance.from.center': x2,
            'player.impact.depth': y2,
            'player.distance.travelled': distance
        }
        data = create_dataset_from_dictionary(data_as_dict)

        res = DistanceTravelledRatio().fit_transform(data)

        assert res[0] == expected
