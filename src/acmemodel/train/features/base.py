"""
Base classes for constructing features.
"""
import re
from typing import List

from sklearn.base import TransformerMixin, BaseEstimator


class BaseFeature(TransformerMixin, BaseEstimator):

    @classmethod
    def name(cls) -> str:
        words = re.findall('[A-Z][^A-Z]*', cls.__name__)
        return '.'.join(words).lower()

    def fit(self, X, y=None) -> BaseEstimator:
        return self

    def get_feature_names(self) -> List[str]:
        return [self.name()]


class ColumnExtractorMixin:

    def transform(self, X):
        assert self._cname is not None, (
            f'_cname is None for {self.__class__.__name__}. '
            f'You need to provide _cname')
        return X[[self._cname]]
