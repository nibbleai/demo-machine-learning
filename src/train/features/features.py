"""
Features
"""
import numpy as np
from scipy.spatial import distance
from sklearn.preprocessing import OneHotEncoder

from .base import BaseFeature, ColumnExtractorMixin


class Speed(BaseFeature, ColumnExtractorMixin):

    _cname = 'speed'


class NetClearance(BaseFeature, ColumnExtractorMixin):

    _cname = 'net.clearance'


class DistanceFromSideline(BaseFeature, ColumnExtractorMixin):

    _cname = 'distance.from.sideline'


class Depth(BaseFeature, ColumnExtractorMixin):

    _cname = 'depth'


class PlayerDistanceTravelled(BaseFeature, ColumnExtractorMixin):

    _cname = 'player.distance.travelled'


class PlayerImpactDepth(BaseFeature, ColumnExtractorMixin):

    _cname = 'player.impact.depth'


class PreviousDistanceFromSideline(BaseFeature, ColumnExtractorMixin):

    _cname = 'previous.distance.from.sideline'


class PreviousTimeToNet(BaseFeature, ColumnExtractorMixin):

    _cname = 'previous.time.to.net'


class Hitpoint(BaseFeature):

    def fit(self, X, y=None):
        encoder = OneHotEncoder(drop='first', sparse=False)
        self.encoder = encoder.fit(X[['hitpoint']])
        return self

    def transform(self, X):
        return self.encoder.transform(X[['hitpoint']])


class Out(BaseFeature, ColumnExtractorMixin):

    def transform(self, X):
        res = X['outside.sideline'] | X['outside.baseline']
        return res.values[:, np.newaxis]


class WeirdNetClearance(BaseFeature):

    def transform(self, X):
        X_tr = (X['net.clearance'] < -0.946) & (X['net.clearance'] > -0.948)
        return X_tr.values[:, np.newaxis]


def distance_travelled_straight_line(row):
    x1 = row['player.distance.from.center']
    y1 = row['player.depth']
    x2 = row['player.impact.distance.from.center']
    y2 = row['player.impact.depth']
    return distance.euclidean((x1, y1), (x2, y2))


class DistanceTravelledRatio(BaseFeature):

    def transform(self, X):
        euclidean_distance = X.apply(distance_travelled_straight_line, axis=1)
        res = np.where(X['player.distance.travelled'] != 0,
                       X['player.distance.travelled'] / euclidean_distance,
                       1)
        return res[:, np.newaxis]
