"""Danger layer and shot calibration.

ShotCalibrator  isotonic map for P(shot), fitted on out-of-match scores; the other three action
                probabilities are rescaled so the four still sum to 1.
DangerModel     P(box entry or shot within the next 5 s of the same team possession), for frames with
                the ball outside the opponent box. Inputs: ball position and xT, best and mean
                teammate xT, ENT_full and its recent trend (1 s ago, 1 s change, 2 s maximum, drop
                from that maximum). Uses past and present only.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .data import FPS, Match
from .models import _booster
from .values import xt_at

HORIZON_S = 5
TREND_FEATURES = ["ent_lag1", "ent_d1", "ent_max2", "ent_drop_from_max"]
DANGER_FEATURES = ["bx", "by", "aby", "xt_ball", "max_mate_xt", "mean_mate_xt", "ent_full"] + TREND_FEATURES
DANGER_PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100, subsample=0.8,
                     subsample_freq=1, colsample_bytree=0.9, verbose=-1, n_jobs=2)


def in_box(x, y, L):
    return (x > L / 2 - 16.5) & (np.abs(y) < 20.16)


class ShotCalibrator:
    def __init__(self, x=None, y=None):
        self.x = None if x is None else np.asarray(x, float)
        self.y = None if y is None else np.asarray(y, float)

    def fit(self, p_shot, is_shot) -> "ShotCalibrator":
        iso = IsotonicRegression(y_min=1e-4, y_max=0.99, out_of_bounds="clip").fit(p_shot, is_shot)
        self.x, self.y = iso.X_thresholds_, iso.y_thresholds_
        return self

    def __call__(self, P: np.ndarray) -> np.ndarray:
        """P: (n, 4) keep, pass, shot, other. Returns recalibrated (n, 4)."""
        if self.x is None:
            return P
        ps = P[:, 2]; pc = np.interp(ps, self.x, self.y)
        scale = (1 - pc) / np.clip(1 - ps, 1e-9, None)
        Q = P * scale[:, None]; Q[:, 2] = pc
        return Q

    def to_dict(self):
        return dict(x=None if self.x is None else self.x.tolist(), y=None if self.y is None else self.y.tolist())


def trend_features(frames: pd.DataFrame) -> pd.DataFrame:
    """frames: match_id, tp, frame, ent_full on the 0.2 s grid. Adds TREND_FEATURES (past only)."""
    S = frames[["match_id", "tp", "frame", "ent_full"]].copy()
    lag = S.copy(); lag["frame"] += FPS
    S = S.merge(lag.rename(columns={"ent_full": "ent_lag1"}), on=["match_id", "tp", "frame"], how="left")
    S["ent_lag1"] = S.ent_lag1.fillna(S.ent_full)
    S["ent_d1"] = S.ent_full - S.ent_lag1
    S = S.sort_values(["match_id", "tp", "frame"])
    S["ent_max2"] = S.groupby(["match_id", "tp"]).ent_full.transform(lambda s: s.rolling(11, min_periods=1).max())
    S["ent_drop_from_max"] = S.ent_max2 - S.ent_full
    return frames.drop(columns=[c for c in TREND_FEATURES if c in frames], errors="ignore").merge(
        S[["match_id", "frame"] + TREND_FEATURES], on=["match_id", "frame"], how="left")


def ball_features(M: Match, frames: np.ndarray, carrier_slots: np.ndarray, signs: np.ndarray, xt) -> pd.DataFrame:
    L, W = M.pitch_length, M.pitch_width
    bx = M.BX[frames] * signs; by = M.BY[frames] * signs
    cx = M.X[frames, carrier_slots] * signs; cy = M.Y[frames, carrier_slots] * signs
    bx = np.where(np.isnan(bx), cx, bx); by = np.where(np.isnan(by), cy, by)
    return pd.DataFrame(dict(bx=bx, by=by, aby=np.abs(by), xt_ball=xt_at(xt, bx, by, L, W), ball_in_box=in_box(bx, by, L)))


def danger_targets(M: Match, C: pd.DataFrame, pp: pd.DataFrame, horizon_s: int = HORIZON_S) -> pd.DataFrame:
    """box / shot within the horizon in the same team possession (10 fps ball track)."""
    L = M.pitch_length
    tpend = pp.groupby("tp").frame_end.max()
    shots = pp[pp.end_type == "shot"].groupby("tp").frame_end.apply(np.sort).to_dict()
    box = np.zeros(len(C), np.int8); shot = np.zeros(len(C), np.int8)
    for i, (f0, tp, s) in enumerate(zip(C.frame.values.astype(int), C.tp.values, C.sign.values)):
        f1 = min(f0 + horizon_s * FPS, int(tpend.get(tp, f0)), M.nf - 1)
        xs = M.BX[f0 + 1:f1 + 1] * s; ys = M.BY[f0 + 1:f1 + 1] * s
        m = ~np.isnan(xs)
        if m.any():
            box[i] = in_box(xs[m], ys[m], L).any()
        sh = shots.get(tp)
        if sh is not None:
            shot[i] = np.any((sh > f0) & (sh <= f0 + horizon_s * FPS))
    return pd.DataFrame(dict(box5=box, shot5=shot, danger5=((box + shot) > 0).astype(np.int8)))


class DangerModel:
    """LightGBM, then an isotonic map fitted on out-of-match scores (the raw model runs a little hot above 0.6)."""
    features = DANGER_FEATURES

    def __init__(self, booster=None, calib_x=None, calib_y=None):
        self.booster = booster
        self.calib_x = None if calib_x is None else np.asarray(calib_x, float)
        self.calib_y = None if calib_y is None else np.asarray(calib_y, float)

    def calibrate(self, raw, y) -> "DangerModel":
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(raw, y)
        self.calib_x, self.calib_y = iso.X_thresholds_, iso.y_thresholds_
        return self

    def predict_raw(self, F: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(F[self.features])

    def fit(self, D: pd.DataFrame) -> "DangerModel":
        """D: frames with the ball outside the box, DANGER_FEATURES, danger5, tp (for weights)."""
        w = 1.0 / D.groupby("tp").tp.transform("size")
        self.booster = _booster(lgb.LGBMClassifier(**DANGER_PARAMS).fit(D[self.features], D.danger5, sample_weight=w))
        return self

    def predict(self, F: pd.DataFrame) -> np.ndarray:
        p = self.predict_raw(F)
        return p if self.calib_x is None else np.interp(p, self.calib_x, self.calib_y)

    def predict_calibrated(self, raw) -> np.ndarray:
        return np.interp(raw, self.calib_x, self.calib_y)

    def calibration_dict(self):
        return dict(x=None if self.calib_x is None else self.calib_x.tolist(), y=None if self.calib_y is None else self.calib_y.tolist())
