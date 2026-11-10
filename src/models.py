"""Baseline strategies for the Hull Tactical Market Prediction competition.

Every model follows the same small interface:
    model.fit(train_df)              -> learns from past data (features + targets)
    model.predict_position(df)       -> daily position in [0, 2] for the rows of df

`df` must contain the feature columns and `lagged_forward_returns` (see data.add_lagged_returns).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import data as D
from .utils import MAX_POSITION, MIN_POSITION, TRADING_DAYS


def clip_position(pos) -> np.ndarray:
    return np.clip(np.asarray(pos, dtype=float), MIN_POSITION, MAX_POSITION)


def signal_to_position(signal, scale: float = 1.0, center: float = 1.0) -> np.ndarray:
    """Map a standardised signal (mean ~0, std ~1) to a position around `center`.

    signal = +1 std -> center + scale, signal = -1 std -> center - scale, clipped to [0, 2].
    """
    return clip_position(center + scale * np.asarray(signal, dtype=float))


def realised_vol(lagged_returns: pd.Series, window: int) -> pd.Series:
    """Rolling standard deviation of past daily returns (uses only information known today)."""
    return lagged_returns.rolling(window, min_periods=max(5, window // 4)).std()


class BaseModel:
    name = "base"

    def fit(self, df: pd.DataFrame) -> "BaseModel":
        return self

    def predict_position(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class ConstantModel(BaseModel):
    """Hold the same position every day. position=1 is buy & hold."""

    def __init__(self, position: float = 1.0):
        self.position = position
        self.name = f"constant_{position:g}"

    def predict_position(self, df):
        return np.full(len(df), self.position, dtype=float)


class BestConstantModel(BaseModel):
    """Choose the constant position that scored best on the training data."""

    name = "best_constant"

    def __init__(self, grid=None):
        self.grid = grid if grid is not None else np.round(np.arange(0.0, 2.01, 0.1), 2)
        self.position = 1.0

    def fit(self, df):
        from .utils import score_frame

        scores = {p: score_frame(df, np.full(len(df), p)) for p in self.grid}
        self.position = max(scores, key=scores.get)
        return self

    def predict_position(self, df):
        return np.full(len(df), self.position, dtype=float)


class VolTargetModel(BaseModel):
    """Volatility targeting: invest less when recent volatility is high.

    position = target_vol / recent_vol, where target_vol is the long-run market volatility
    measured on the training data. Motivated by the volatility clustering seen in the EDA.
    """

    def __init__(self, window: int = 21, max_position: float = 1.5):
        self.window = window
        self.max_position = max_position
        self.target_vol = None
        self.name = f"vol_target_{window}d"

    def fit(self, df):
        self.target_vol = df["forward_returns"].std()
        return self

    def predict_position(self, df):
        vol = realised_vol(df["lagged_forward_returns"], self.window)
        pos = (self.target_vol / vol).fillna(1.0)
        return clip_position(np.minimum(pos, self.max_position))


class MomentumModel(BaseModel):
    """Simple trend rule: more exposure after a good past year, less after a bad one."""

    def __init__(self, window: int = 252, high: float = 1.2, low: float = 0.6):
        self.window, self.high, self.low = window, high, low
        self.name = f"momentum_{window}d"

    def predict_position(self, df):
        past = df["lagged_forward_returns"].rolling(self.window, min_periods=self.window // 2).sum()
        pos = np.where(past > 0, self.high, self.low)
        pos[past.isna().values] = 1.0
        return clip_position(pos)


class LinearModel(BaseModel):
    """Regularised linear regression on all features, predicting next-day excess return.

    The prediction is standardised with the spread of in-sample predictions and mapped to a position
    with `signal_to_position`.
    """

    def __init__(self, kind: str = "ridge", alpha: float = 10.0, l1_ratio: float = 0.5, scale: float = 0.5):
        self.kind, self.alpha, self.l1_ratio, self.scale = kind, alpha, l1_ratio, scale
        self.name = f"{kind}"
        self.features = None
        self.medians = None
        self.pipe = None
        self.pred_mean = 0.0
        self.pred_std = 1.0

    def _make_estimator(self):
        if self.kind == "ridge":
            est = Ridge(alpha=self.alpha)
        elif self.kind == "elasticnet":
            est = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=10_000)
        else:
            raise ValueError(self.kind)
        return make_pipeline(StandardScaler(), est)

    def fit(self, df):
        self.features = D.feature_columns(df)
        X, self.medians = D.fill_missing(df[self.features])
        y = df[D.TARGET].values
        self.pipe = self._make_estimator().fit(X, y)
        pred = self.pipe.predict(X)
        self.pred_mean, self.pred_std = pred.mean(), pred.std() or 1.0
        return self

    def predict_signal(self, df):
        X, _ = D.fill_missing(df[self.features], self.medians)
        return (self.pipe.predict(X) - self.pred_mean) / self.pred_std

    def predict_position(self, df):
        return signal_to_position(self.predict_signal(df), scale=self.scale)

    def coefficients(self) -> pd.Series:
        coef = self.pipe[-1].coef_
        return pd.Series(coef, index=self.features).sort_values(key=np.abs, ascending=False)


class LightGBMModel(LinearModel):
    """Gradient-boosted trees (LightGBM) on all features, predicting next-day excess return.

    Same signal -> position mapping as LinearModel. Small, heavily regularised trees because the
    signal is weak and noisy (see EDA). Trees handle missing values natively, so no imputation.
    """

    def __init__(self, scale: float = 0.5, n_estimators: int = 300, learning_rate: float = 0.02,
                 num_leaves: int = 15, min_child_samples: int = 200, seed: int = 42):
        self.scale = scale
        self.params = dict(n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves,
                           min_child_samples=min_child_samples, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.7, reg_lambda=5.0, random_state=seed, verbose=-1)
        self.name = "lightgbm"
        self.features = None
        self.model = None
        self.pred_mean, self.pred_std = 0.0, 1.0

    def fit(self, df):
        from lightgbm import LGBMRegressor

        self.features = D.feature_columns(df)
        self.model = LGBMRegressor(**self.params).fit(df[self.features], df[D.TARGET])
        pred = self.model.predict(df[self.features])
        self.pred_mean, self.pred_std = pred.mean(), pred.std() or 1.0
        return self

    def predict_signal(self, df):
        return (self.model.predict(df[self.features]) - self.pred_mean) / self.pred_std

    def feature_importance(self) -> pd.Series:
        imp = self.model.booster_.feature_importance(importance_type="gain")
        return pd.Series(imp, index=self.features).sort_values(ascending=False)


class OnlinePredictor:
    """Wraps a fitted model for Kaggle's evaluation API, which sends one day at a time.

    Rolling-window models (volatility targeting, momentum) need recent history, so the predictor keeps
    the training rows plus every test row it has seen and predicts the position for the newest day.
    """

    def __init__(self, model: BaseModel, train_df: pd.DataFrame, history: int = 600):
        self.model = model
        self.history = history
        cols = [c for c in train_df.columns if c not in D.LEAK_COLS]
        self.buffer = train_df[cols].copy()

    def predict_one(self, test_row: pd.DataFrame) -> float:
        test_row = test_row.copy()
        date_id = int(test_row["date_id"].iloc[0])
        past = self.buffer[self.buffer["date_id"] < date_id]
        frame = pd.concat([past.tail(self.history), test_row[past.columns.intersection(test_row.columns)]],
                          ignore_index=True)
        self.buffer = pd.concat([past, frame.tail(1)], ignore_index=True)
        pos = float(self.model.predict_position(frame)[-1])
        return float(np.clip(pos, MIN_POSITION, MAX_POSITION))


def baseline_zoo() -> dict[str, BaseModel]:
    """All Assignment 1 baselines with their default settings."""
    models = [
        ConstantModel(1.0),
        BestConstantModel(),
        VolTargetModel(window=21),
        VolTargetModel(window=63),
        MomentumModel(window=252),
        LinearModel("ridge", alpha=1000.0),
        LinearModel("elasticnet", alpha=1e-4, l1_ratio=0.5),
        LightGBMModel(),
    ]
    return {m.name: m for m in models}


__all__ = [
    "ConstantModel", "BestConstantModel", "VolTargetModel", "MomentumModel", "LinearModel", "LightGBMModel",
    "OnlinePredictor",
    "baseline_zoo", "signal_to_position", "clip_position", "TRADING_DAYS",
]
