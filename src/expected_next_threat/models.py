"""The three learned models. Each probability model gets a temperature fitted on inner,
match-grouped out-of-fold scores, so its probabilities stay honest on unseen matches.

ReceiverModel  P(receiver = j | pass) over all outfield teammates (per-candidate LightGBM + softmax)
ActionModel    P(keep, pass, shot, other) in the next 1 s (LightGBM multiclass)
CarryModel     1 s displacement of the carrier (mean and spread), for the expected carry value
"""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from .features import ACTION_FEATURES, CARRIER_FEATURES, RECEIVER_FEATURES

RECEIVER_PARAMS = dict(n_estimators=250, learning_rate=0.06, num_leaves=31, min_child_samples=40, subsample=0.8,
                       subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, n_jobs=2)
ACTION_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=100, subsample=0.8, subsample_freq=1,
                     colsample_bytree=0.8, verbose=-1, n_jobs=2)
CLASSES = ["keep", "pass", "shot", "other"]


def group_softmax(z, g):
    """Softmax of scores z within groups g (integer codes)."""
    z = z - pd.Series(z).groupby(g).transform("max").values
    e = np.exp(z)
    return e / np.bincount(g, weights=e)[g]


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def _booster(m):
    return m.booster_ if hasattr(m, "booster_") else m


class ReceiverModel:
    features = RECEIVER_FEATURES

    def __init__(self, booster=None, temperature=1.0):
        self.booster = booster; self.temperature = float(temperature)

    def fit(self, C: pd.DataFrame, n_inner: int = 3) -> "ReceiverModel":
        """C: candidate rows with features, y (1 = targeted), w (pass weight), frame_uid, match_id."""
        ms = np.sort(C.match_id.unique())
        inner = np.array_split(ms, n_inner); oof = np.zeros(len(C))
        for im in inner:
            it = C.match_id.isin(im).values
            mm = lgb.LGBMClassifier(**RECEIVER_PARAMS).fit(C.loc[~it, self.features], C.y[~it], sample_weight=C.w[~it])
            oof[it] = mm.predict(C.loc[it, self.features], raw_score=True)
        gt = pd.factorize(C.frame_uid)[0]; yt = C.y.values; wt = C.w.values; pos = yt == 1

        def nll(a):
            p = group_softmax(a * oof, gt)
            return -np.average(np.log(np.clip(p[pos], 1e-12, 1)), weights=wt[pos])
        self.temperature = float(minimize_scalar(nll, bounds=(0.2, 3), method="bounded").x)
        self.booster = _booster(lgb.LGBMClassifier(**RECEIVER_PARAMS).fit(C[self.features], C.y, sample_weight=C.w))
        return self

    def predict(self, Cf: pd.DataFrame, groups: np.ndarray) -> np.ndarray:
        return group_softmax(self.temperature * self.booster.predict(Cf[self.features], raw_score=True), groups)


class ActionModel:
    features = ACTION_FEATURES

    def __init__(self, booster=None, temperature=1.0):
        self.booster = booster; self.temperature = float(temperature)

    def fit(self, A: pd.DataFrame, n_inner: int = 3) -> "ActionModel":
        """A: carrier rows with features and y in 0..3 (keep, pass, shot, other)."""
        ms = np.sort(A.match_id.unique())
        inner = np.array_split(ms, n_inner); oof = np.zeros((len(A), 4))
        for im in inner:
            it = A.match_id.isin(im).values
            m = lgb.LGBMClassifier(objective="multiclass", **ACTION_PARAMS).fit(A.loc[~it, self.features], A.y[~it])
            oof[it] = m.predict(A.loc[it, self.features], raw_score=True)
        y = A.y.values
        self.temperature = float(minimize_scalar(lambda a: -np.log(softmax(a * oof)[np.arange(len(y)), y] + 1e-12).mean(),
                                                 bounds=(0.3, 3), method="bounded").x)
        self.booster = _booster(lgb.LGBMClassifier(objective="multiclass", **ACTION_PARAMS).fit(A[self.features], A.y))
        return self

    def predict(self, F: pd.DataFrame) -> np.ndarray:
        return softmax(self.temperature * self.booster.predict(F[self.features], raw_score=True))


class CarryModel:
    """Mean and spread of the 1 s displacement (dx forward, dy outward). sd = 1.2533 x E|residual|."""
    features = CARRIER_FEATURES
    targets = ("dx", "dy_out")

    def __init__(self, boosters=None):
        self.boosters = boosters or {}

    def fit(self, K: pd.DataFrame, n_inner: int = 3) -> "CarryModel":
        ms = np.sort(K.match_id.unique())
        for tgt in self.targets:
            inner = np.array_split(ms, n_inner); res = np.zeros(len(K))
            for im in inner:
                it = K.match_id.isin(im).values
                res[it] = K[tgt].values[it] - lgb.LGBMRegressor(**ACTION_PARAMS).fit(K.loc[~it, self.features], K[tgt][~it]).predict(K.loc[it, self.features])
            self.boosters[tgt] = (_booster(lgb.LGBMRegressor(**ACTION_PARAMS).fit(K[self.features], K[tgt])),
                                  _booster(lgb.LGBMRegressor(**ACTION_PARAMS).fit(K[self.features], np.abs(res))))
        return self

    def predict(self, F: pd.DataFrame):
        out = {}
        for tgt in self.targets:
            mu, sd = self.boosters[tgt]
            out[tgt] = (mu.predict(F[self.features]), np.maximum(1.2533 * sd.predict(F[self.features]), 0.2))
        return out


def save_booster(b, path: Path):
    b.save_model(str(path))


def load_booster(path: Path):
    return lgb.Booster(model_file=str(path))
