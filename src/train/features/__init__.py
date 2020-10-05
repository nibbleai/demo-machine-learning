from sklearn.pipeline import FeatureUnion

from .features import (
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

FEATURES_LIST = [
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
]

FEATURES_STORE = [(f.name(), f()) for f in FEATURES_LIST]

features_generator = FeatureUnion(FEATURES_STORE)
