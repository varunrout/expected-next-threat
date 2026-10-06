"""Features. Every feature at frame f uses only frames f, f - 0.5 s and f - 1 s.

Coordinates are flipped by `sign` so the team in possession attacks +x.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import FPS

RECEIVER_POS = ["d", "dxr", "adyr", "xj", "ayj", "dgoal_j", "rank_d", "d_rel_min", "n_cand",
                "nd_j", "nd3_j", "lane_min", "lane_n2", "xp", "ayp"]
RECEIVER_MOVE = ["vxj", "vyj_out", "spj", "vrad_j", "vlat_j", "vpj_to", "vp_sp", "vp_cos"]
RECEIVER_FEATURES = RECEIVER_POS + RECEIVER_MOVE + ["t_on_ball"]

CARRIER_FEATURES = ["cx", "acy", "dgoal", "ang_goal", "vx1", "vy1_out", "sp1", "vx5", "vy5_out", "sp5", "acc", "t_on_ball",
                    "d1", "d2", "close_v", "n3", "n5", "n10", "cone_d", "goal_side", "mates_ahead"]
ACTION_FEATURES = CARRIER_FEATURES + ["xg_here", "xt_here", "pmax", "ent", "max_mate_xt", "mean_mate_xt"]
BALL_SPEED = 15.0  # m/s, for the receiver's target point


def candidate_features(M, frames, pslots, teams, signs, L, W, chunk=20000) -> pd.DataFrame:
    """One row per (frame, outfield teammate of the carrier). `row` indexes the input arrays."""
    out = []
    for a in range(0, len(frames), chunk):
        f = frames[a:a + chunk]; p = pslots[a:a + chunk]; t = teams[a:a + chunk]; s = signs[a:a + chunk][:, None]
        n = len(f)
        X = M.X[f] * s; Y = M.Y[f] * s
        fl1 = np.maximum(f - FPS, 0); fl5 = np.maximum(f - FPS // 2, 0)
        VX = (X - M.X[fl1] * s); VY = (Y - M.Y[fl1] * s)
        r = np.arange(n)
        xp = X[r, p]; yp = Y[r, p]
        vpx = (xp - M.X[fl5, p] * s[:, 0]) * 2; vpy = (yp - M.Y[fl5, p] * s[:, 0]) * 2
        present = ~np.isnan(X)
        mate = (M.team[None, :] == t[:, None]) & ~M.gk[None, :] & present
        mate[r, p] = False
        opp = (M.team[None, :] != t[:, None]) & present
        dxr = X - xp[:, None]; dyr = Y - yp[:, None]
        d = np.hypot(dxr, dyr)
        ux = dxr / np.maximum(d, 0.1); uy = dyr / np.maximum(d, 0.1)
        DX = X[:, :, None] - np.where(opp, X, np.nan)[:, None, :]
        DY = Y[:, :, None] - np.where(opp, Y, np.nan)[:, None, :]
        DD = np.hypot(DX, DY)
        nd = np.nanmin(np.where(np.isnan(DD), np.inf, DD), axis=2)
        nd3 = np.nansum(DD < 3, axis=2)
        ox = np.where(opp, X, np.nan)[:, None, :] - xp[:, None, None]
        oy = np.where(opp, Y, np.nan)[:, None, :] - yp[:, None, None]
        seg2 = np.maximum(d ** 2, 0.01)[:, :, None]
        tt = np.clip((ox * dxr[:, :, None] + oy * dyr[:, :, None]) / seg2, 0, 1)
        px_ = tt * dxr[:, :, None] - ox; py_ = tt * dyr[:, :, None] - oy
        perp = np.hypot(px_, py_)
        perp = np.where((tt > 0.05) & (tt < 0.98), perp, np.inf)
        lane_min = np.min(np.where(np.isnan(perp), np.inf, perp), axis=2)
        lane_n2 = np.sum(perp < 2, axis=2)
        dm = np.where(mate, d, np.inf)
        rank = np.argsort(np.argsort(dm, axis=1), axis=1) + 1
        dmin = dm.min(1, keepdims=True)
        vrad = VX * ux + VY * uy
        vlat = np.abs(-VX * uy + VY * ux)
        vp_to = vpx[:, None] * ux + vpy[:, None] * uy
        vp_sp = np.hypot(vpx, vpy)
        vp_cos = vp_to / np.maximum(vp_sp[:, None], 0.3)
        ii, jj = np.nonzero(mate)
        g = lambda A: A[ii, jj]
        out.append(pd.DataFrame(dict(
            row=ii + a, slot=jj, d=g(d), dxr=g(dxr), adyr=np.abs(g(dyr)), xj=g(X), ayj=np.abs(g(Y)),
            dgoal_j=np.hypot(L / 2 - g(X), g(Y)), rank_d=g(rank), d_rel_min=g(d) / np.maximum(dmin[ii, 0], 0.5),
            n_cand=mate.sum(1)[ii], nd_j=np.minimum(g(nd), 40), nd3_j=g(nd3), lane_min=np.minimum(g(lane_min), 30),
            lane_n2=g(lane_n2), xp=xp[ii], ayp=np.abs(yp[ii]),
            vxj=g(VX), vyj_out=g(VY) * np.sign(g(Y) + 1e-6), spj=np.hypot(g(VX), g(VY)), vrad_j=g(vrad), vlat_j=g(vlat),
            vpj_to=g(vp_to), vp_sp=vp_sp[ii], vp_cos=g(vp_cos),
            det_j=M.DET[f[ii], jj], yj=g(Y))))
    if not out:
        return pd.DataFrame(columns=["row", "slot"] + RECEIVER_POS + RECEIVER_MOVE + ["det_j", "yj"])
    return pd.concat(out, ignore_index=True)


def receiver_targets(M, fj, slot, cs, sj, L, W, iters=2):
    """Where each candidate will be when a pass arrives: position + velocity x ball time (capped 2 s)."""
    X = M.X[fj, slot] * sj; Y = M.Y[fj, slot] * sj
    VX = np.nan_to_num(X - M.X[np.maximum(fj - FPS, 0), slot] * sj); VY = np.nan_to_num(Y - M.Y[np.maximum(fj - FPS, 0), slot] * sj)
    BX = M.BX[fj] * sj; BY = M.BY[fj] * sj
    BX = np.where(np.isnan(BX), M.X[fj, cs] * sj, BX); BY = np.where(np.isnan(BY), M.Y[fj, cs] * sj, BY)
    tb = np.hypot(X - BX, Y - BY) / BALL_SPEED
    for _ in range(iters):
        tx = np.clip(X + VX * np.minimum(tb, 2), -L / 2, L / 2); ty = np.clip(Y + VY * np.minimum(tb, 2), -W / 2, W / 2)
        tb = np.hypot(tx - BX, ty - BY) / BALL_SPEED
    return tx, ty


def carrier_features(M, f, cs, tm, s, tob, L, W, xt, xg) -> pd.DataFrame:
    """One row per carrier frame: position, movement, pressure and context (no pass-layer columns)."""
    from .values import xt_at
    n = len(f); r = np.arange(n)
    Xa = M.X[f] * s[:, None]; Ya = M.Y[f] * s[:, None]
    X1 = M.X[np.maximum(f - FPS, 0)] * s[:, None]; Y1 = M.Y[np.maximum(f - FPS, 0)] * s[:, None]
    X5 = M.X[np.maximum(f - FPS // 2, 0)] * s[:, None]; Y5 = M.Y[np.maximum(f - FPS // 2, 0)] * s[:, None]
    cx, cy = Xa[r, cs], Ya[r, cs]
    vx1, vy1 = cx - X1[r, cs], cy - Y1[r, cs]
    vx5, vy5 = (cx - X5[r, cs]) * 2, (cy - Y5[r, cs]) * 2
    present = ~np.isnan(Xa)
    opp = (M.team[None, :] != tm[:, None]) & present
    mate = (M.team[None, :] == tm[:, None]) & present & ~M.gk[None, :]
    mate[r, cs] = False
    dd = np.where(opp, np.hypot(Xa - cx[:, None], Ya - cy[:, None]), np.inf)
    ds = np.sort(dd, axis=1); k1 = np.argmin(dd, axis=1)
    dd_prev = np.hypot(X5[r, k1] - (cx - vx5 / 2), Y5[r, k1] - (cy - vy5 / 2))
    rx, ry = Xa - cx[:, None], Ya - cy[:, None]
    cone = opp & (rx > 0) & (np.abs(ry) <= rx)
    return pd.DataFrame(dict(
        cx=cx, cy=cy, acy=np.abs(cy), dgoal=np.hypot(L / 2 - cx, cy), ang_goal=np.abs(np.arctan2(cy, L / 2 - cx)),
        vx1=vx1, vy1_out=vy1 * np.sign(cy + 1e-6), sp1=np.hypot(vx1, vy1), vx5=vx5, vy5_out=vy5 * np.sign(cy + 1e-6),
        sp5=np.hypot(vx5, vy5), acc=np.hypot(vx5, vy5) - np.hypot(vx1, vy1), t_on_ball=tob,
        d1=np.minimum(ds[:, 0], 40), d2=np.minimum(ds[:, 1], 40), close_v=(dd_prev - ds[:, 0]) * 2,
        n3=(dd < 3).sum(1), n5=(dd < 5).sum(1), n10=(dd < 10).sum(1),
        cone_d=np.minimum(np.where(cone, np.hypot(rx, ry), np.inf).min(1), 40),
        goal_side=(opp & (Xa > cx[:, None])).sum(1), mates_ahead=(mate & (Xa > cx[:, None])).sum(1),
        xg_here=xg(cx, cy, L), xt_here=xt_at(xt, cx, cy, L, W), det_c=M.DET[f, cs], det_d1=M.DET[f, k1]))
