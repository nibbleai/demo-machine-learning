from json.encoder import JSONEncoder as _JSONEncoder
from typing import Any

from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    TransformerMixin
)


class JSONEncoder(_JSONEncoder):
    """A custom JSON encoder that allows encoding scikit-learn's Pipelines."""
    def default(self, o: Any) -> Any:
        if isinstance(o, ClassifierMixin) and isinstance(o, BaseEstimator):
            return {
                'classifier': _get_dict_representation(o)
            }
        elif isinstance(o, RegressorMixin) and isinstance(o, BaseEstimator):
            return {
                'regressor': _get_dict_representation(o)
            }
        elif isinstance(o, TransformerMixin) and isinstance(o, BaseEstimator):
            return {
                'transformer': _get_dict_representation(o)
            }
        elif isinstance(o, BaseEstimator):
            return {
                'estimator': _get_dict_representation(o)
            }
        elif hasattr(o, '__name__'):
            return o.__name__
        else:
            try:
                return super().default(o)
            except TypeError:  # ... is not JSON serializable
                if hasattr(o, '__dict__'):
                    return o.__dict__

        # We tried everything we could, but did not manage to get
        # a proper JSON reprensation of the object
        return None


def _get_dict_representation(o: BaseEstimator) -> dict:
    return {
        'type': o.__class__.__name__,
        'params': o.get_params()
    }
