"""Preprocessing fitted on the training period only.

The scaler and the categorical vocabulary are fitted on train rows and then
frozen. Fitting on anything later - validation, backtest or forward - would let
future information influence the model, so ``fit`` accepts the training frame
explicitly and refuses a frame that reaches past the training cutoff.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from driftguard.data.schema import CATEGORICAL_FIELDS, FlowFrame
from driftguard.temporal import TemporalLeakageError

# Never model on: the target, the time index, or any identity field.
FORBIDDEN = {"label", "timestamp"}


class FeaturePreprocessor:
    """Numeric scaling plus one-hot encoding, both fitted on train only."""

    def __init__(self, numeric_features: Sequence[str], categorical_features: Sequence[str] = ()):
        self.numeric_features = [f for f in numeric_features if f not in FORBIDDEN]
        self.categorical_features = [f for f in categorical_features if f not in FORBIDDEN]
        self.scaler = StandardScaler()
        self.categories_: Dict[str, List] = {}
        self.fitted_ = False

    def _check_policy(self, feature_set: Sequence[str]) -> None:
        illegal = [f for f in feature_set if f in FORBIDDEN]
        if illegal:
            raise ValueError(f"feature policy violation: {illegal} must never be model inputs")

    def fit(self, frame: FlowFrame, cutoff: Optional[pd.Timestamp] = None) -> "FeaturePreprocessor":
        if cutoff is not None:
            if frame.timestamps.max() > cutoff:
                raise TemporalLeakageError(
                    f"preprocessor fitted on rows up to {frame.timestamps.max()}, past the training cutoff {cutoff}"
                )
        self._check_policy(self.numeric_features + self.categorical_features)
        numeric = frame.frame[self.numeric_features].to_numpy(dtype=float)
        self.scaler.fit(numeric)
        for column in self.categorical_features:
            values = frame.frame[column].astype(str)
            self.categories_[column] = sorted(values.unique().tolist())
        self.fitted_ = True
        return self

    def transform(self, frame: FlowFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("FeaturePreprocessor.transform called before fit")
        numeric = frame.frame[self.numeric_features].to_numpy(dtype=float)
        scaled = self.scaler.transform(numeric)
        out = pd.DataFrame(scaled, columns=self.numeric_features, index=frame.frame.index)
        for column in self.categorical_features:
            known = self.categories_[column]
            values = frame.frame[column].astype(str)
            indicator = np.zeros((len(values), len(known)), dtype=float)
            lookup = {name: i for i, name in enumerate(known)}
            for row_index, value in enumerate(values):
                # Unseen categories fall into an all-zero row rather than
                # inventing a new column that the scaler never saw.
                if value in lookup:
                    indicator[row_index, lookup[value]] = 1.0
            out[[f"{column}={name}" for name in known]] = indicator
        return out

    def fit_transform(self, frame: FlowFrame, cutoff: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        return self.fit(frame, cutoff=cutoff).transform(frame)

    def output_features(self) -> List[str]:
        columns = list(self.numeric_features)
        for column in self.categorical_features:
            columns += [f"{column}={name}" for name in self.categories_.get(column, [])]
        return columns

    def provenance(self) -> Dict[str, object]:
        return {
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "scaler_mean": [round(float(v), 6) for v in self.scaler.mean_],
            "categories": self.categories_,
        }


def drop_feature_groups(features: Sequence[str], groups: Sequence[str], group_map: Dict[str, Sequence[str]]) -> List[str]:
    """Remove the named feature groups, for the ablation studies."""
    removed = set()
    for group in groups:
        if group not in group_map:
            raise KeyError(f"unknown feature group '{group}'. Available: {sorted(group_map)}")
        removed.update(group_map[group])
    return [f for f in features if f not in removed]
