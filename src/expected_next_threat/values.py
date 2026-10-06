"""Value maps: expected threat (xT) on a 12 x 8 grid and a distance-angle xG model.

xT follows Singh's Markov formulation refitted on SkillCorner player possessions. Moves are
normalised by every move attempt from a cell (passes of any outcome, losses, clearances, carries),
so failed moves are not counted as completions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

NX, NY = 12, 8
_POST = 7.32


def _cell(x, y, L, W):
    cx = np.clip(((np.asarray(x) + L / 2) / L * NX).astype(int), 0, NX - 1)
    cy = np.clip(((np.asarray(y) + W / 2) / W * NY).astype(int), 0, NY - 1)
    return cx * NY + cy


def xt_at(xt: np.ndarray, x, y, L, W) -> np.ndarray:
    """xT at attack-normalised positions (team attacks +x; centre origin, metres)."""
    return xt[_cell(x, y, L, W)]


def _dist_angle(x, y, L):
    dx = L / 2 - np.asarray(x, float); y = np.asarray(y, float)
    return np.hypot(dx, y), np.abs(np.arctan2(_POST * dx, dx ** 2 + y ** 2 - (_POST / 2) ** 2))


def fit_xt(pp: pd.DataFrame, pitch: pd.DataFrame, iters: int = 50) -> np.ndarray:
    """pp: stitched player possessions (all matches); pitch: index match_id, pitch_length, pitch_width."""
    pp = pp.join(pitch[["pitch_length", "pitch_width"]], on="match_id")
    nxt = pp.groupby(["match_id", "period"]).shift(-1)
    L, W = pp.pitch_length.values, pp.pitch_width.values
    c_end = _cell(pp.x_end.values, pp.y_end.values, L, W)
    succ = (nxt.tp == pp.tp).values & (pp.end_type == "pass").values
    c_dest = _cell(nxt.x_start.fillna(0).values, nxt.y_start.fillna(0).values, L, W)
    is_shot = (pp.end_type == "shot").values
    is_goal = is_shot & pp.lead_to_goal.fillna(False).values.astype(bool)
    N = NX * NY
    cnt = np.bincount(c_end, minlength=N).astype(float)
    shot_n = np.bincount(c_end[is_shot], minlength=N).astype(float)
    T = np.zeros((N, N))
    np.add.at(T, (c_end[succ], c_dest[succ]), 1)
    c_start = _cell(pp.x_start.values, pp.y_start.values, L, W)
    carry_n = np.bincount(c_start, minlength=N).astype(float)
    np.add.at(T, (c_start, c_end), 1)
    tot = cnt + carry_n
    p_shot = np.divide(shot_n, tot, out=np.zeros(N), where=tot > 0)
    p_move = 1 - p_shot
    sx, sy = pp.x_end.values[is_shot], pp.y_end.values[is_shot]
    half = L[is_shot] / 2
    dist = np.hypot(half - sx, sy)
    ang = np.abs(np.arctan2(_POST * (half - sx), (half - sx) ** 2 + sy ** 2 - (_POST / 2) ** 2))
    lr = LogisticRegression().fit(np.c_[dist, ang], is_goal[is_shot])
    Lm, Wm = 105.0, 68.0
    cxs = (np.arange(NX) + .5) / NX * Lm - Lm / 2
    cys = (np.arange(NY) + .5) / NY * Wm - Wm / 2
    CX, CY = np.meshgrid(cxs, cys, indexing="ij")
    cd = np.hypot(Lm / 2 - CX.ravel(), CY.ravel())
    ca = np.abs(np.arctan2(_POST * (Lm / 2 - CX.ravel()), (Lm / 2 - CX.ravel()) ** 2 + CY.ravel() ** 2 - (_POST / 2) ** 2))
    p_goal = lr.predict_proba(np.c_[cd, ca])[:, 1]
    moves = (tot - shot_n)[:, None]
    Tm = np.divide(T, moves, out=np.zeros_like(T), where=moves > 0)
    xt = np.zeros(N)
    for _ in range(iters):
        xt = p_shot * p_goal + p_move * (Tm @ xt)
    return xt


class XG:
    """P(goal | shot) from distance and angle to goal. Stores only three numbers."""

    def __init__(self, coef=None, intercept=None):
        self.coef = None if coef is None else np.asarray(coef, float)
        self.intercept = None if intercept is None else float(intercept)

    def fit(self, pp: pd.DataFrame, pitch: pd.DataFrame) -> "XG":
        pp = pp.join(pitch[["pitch_length"]], on="match_id")
        sh = pp[pp.end_type == "shot"]
        d, a = _dist_angle(sh.x_end.values, sh.y_end.values, sh.pitch_length.values)
        lr = LogisticRegression().fit(np.c_[d, a], sh.lead_to_goal.fillna(False).astype(bool).values)
        self.coef = lr.coef_[0]; self.intercept = float(lr.intercept_[0])
        return self

    def __call__(self, x, y, L) -> np.ndarray:
        d, a = _dist_angle(x, y, L)
        z = self.intercept + self.coef[0] * d + self.coef[1] * a
        return 1.0 / (1.0 + np.exp(-z))

    def to_dict(self):
        return dict(coef=self.coef.tolist(), intercept=self.intercept)
