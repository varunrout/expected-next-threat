"""Build training rows and fit the engine.

The pass layer that feeds the action model is cross-fitted: each match's receiver probabilities
come from a receiver model that never saw that match, exactly as at prediction time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .danger import DANGER_FEATURES, TREND_FEATURES, DangerModel, ShotCalibrator, ball_features, danger_targets, trend_features
from types import SimpleNamespace

from .data import FPS, MATCH_IDS, Match, load_match
from .engine import NextThreatEngine
from .features import candidate_features, carrier_features
from .models import CLASSES, ActionModel, CarryModel, ReceiverModel
from .possessions import carrier_frames, stitch_possessions
from .values import XG, fit_xt

HORIZON = FPS  # 1 s
# match folds used for every out-of-match number in the analysis (fixed for reproducibility)
FOLDS = ((1959846, 2016236, 1986691, 2006229), (1899585, 2006363, 1874553, 2015213), (2017461, 2013725, 1996436, 2007448),
         (1953632, 2011166, 2007721, 2010085), (1925299, 1927964, 1996435, 1886347))


def receiver_rows(M: Match, pp: pd.DataFrame, max_lag: int = 40, step: int = 2) -> pd.DataFrame:
    """Every 0.2 s carrier frame of each open-play targeted pass, from reception to 0.2 s before release
    (at most 4 s). Each pass is weighted once."""
    L, W = M.pitch_length, M.pitch_width
    g = pp[(pp.match_id == M.match_id) & (pp.end_type == "pass") & pp.player_targeted_id.notna()
           & pp.game_interruption_before.isna() & (pp.team_in_possession_phase_type != "set_play")]
    g = g[g.player_id.isin(M.slot) & g.player_targeted_id.isin(M.slot)].reset_index(drop=True)
    fr, ev, lg = [], [], []
    for i, r in enumerate(g.itertuples()):
        fe = int(r.frame_end); lo = max(int(r.frame_start), fe - max_lag)
        fs = np.arange(fe - 2, lo - 1, -step)
        if len(fs) == 0:
            fs = np.array([fe - 2])
        fs = fs[(fs >= FPS) & (fs < M.nf)]
        fr.append(fs); ev.append(np.full(len(fs), i)); lg.append(fe - fs)
    fr = np.concatenate(fr).astype(int); ev = np.concatenate(ev).astype(int); lg = np.concatenate(lg)
    ps_ = g.player_id.map(M.slot).values[ev]
    sg = np.where(g.attacking_side.values == "left_to_right", 1.0, -1.0)[ev]
    Cf = candidate_features(M, fr, ps_, g.team_id.values[ev], sg, L, W)
    rr = Cf.row.values; e = ev[rr]
    tgt = g.player_targeted_id.map(M.slot).values
    Cf["y"] = (Cf.slot.values == tgt[e]).astype(int)
    Cf["pass_uid"] = M.match_id * 100000 + g.event_id.values[e].astype(np.int64)
    Cf["frame_uid"] = Cf.pass_uid * 100 + lg[rr] // step
    Cf["lag"] = lg[rr] / FPS
    Cf["t_on_ball"] = (fr[rr] - g.frame_start.values[e]) / FPS
    Cf["match_id"] = M.match_id
    return Cf.drop(columns="row")


def finish_receiver_rows(C: pd.DataFrame) -> pd.DataFrame:
    C = C[C.groupby("frame_uid").y.transform("sum") == 1].reset_index(drop=True)
    C["w"] = 1.0 / C.groupby("pass_uid").frame_uid.transform("nunique")
    return C


def action_rows(M: Match, C: pd.DataFrame, agg: pd.DataFrame, xt, xg) -> pd.DataFrame:
    """Carrier rows with pass-layer summaries, the 1 s action label and the 1 s carry displacement."""
    ok = agg.ent.notna().values  # frames with at least one candidate teammate
    C = C[ok].reset_index(drop=True); agg = agg[ok].reset_index(drop=True)
    f = C.frame.values.astype(int); cs = C.carrier_slot.values.astype(int); s = C.sign.values.astype(float)
    F = carrier_features(M, f, cs, C.team_id.values, s, C.t_on_ball.values, M.pitch_length, M.pitch_width, xt, xg)
    for c in ("ent", "pmax", "max_mate_xt", "mean_mate_xt"):
        F[c] = agg[c].values
    tte = C.frame_end.values.astype(int) - f
    et = C.end_type.values
    act = np.where(tte > HORIZON, "keep", np.where(et == "pass", "pass", np.where(et == "shot", "shot", "other")))
    fk = np.minimum(f + HORIZON, M.nf - 1)
    fx = np.where(act == "keep", M.X[fk, cs] * s, np.nan); fy = np.where(act == "keep", M.Y[fk, cs] * s, np.nan)
    F["dx"] = fx - F.cx.values; F["dy_out"] = (fy - F.cy.values) * np.sign(F.cy.values + 1e-6)
    F["act"] = act; F["y"] = pd.Series(act).map({c: i for i, c in enumerate(CLASSES)}).values
    F.insert(0, "match_id", M.match_id); F.insert(1, "frame", f); F.insert(2, "tp", C.tp.values)
    F["tte"] = tte / FPS
    return F


def prepare(match_ids=MATCH_IDS, quiet=True):
    """Load matches, stitch possessions, fit the value maps."""
    ms = {m: load_match(m, quiet=quiet) for m in match_ids}
    ev = pd.concat([m.events for m in ms.values()], ignore_index=True)
    pp, tp = stitch_possessions(ev)
    pitch = pd.DataFrame({m: dict(pitch_length=M.pitch_length, pitch_width=M.pitch_width) for m, M in ms.items()}).T
    xt = fit_xt(pp, pitch)
    xg = XG().fit(pp, pitch)
    return ms, pp, xt, xg


def fit_engine(match_ids=MATCH_IDS, folds=FOLDS, quiet=True, log=print):
    """Fit the full engine on match_ids, and score every match with fold models that never saw it.

    Returns a namespace: engine (fitted on all matches), oof (out-of-match scored carrier frames),
    A (action rows), R (receiver rows with out-of-match p_oof), ms, pp, carriers, folds."""
    ms, pp, xt, xg = prepare(match_ids, quiet)
    folds = [tuple(m for m in fm if m in ms) for fm in folds]
    R = finish_receiver_rows(pd.concat([receiver_rows(ms[m], pp) for m in match_ids], ignore_index=True))
    carriers = {m: carrier_frames(ms[m], pp, drop_untracked=False) for m in match_ids}  # as in the analysis
    rec_folds, A_parts = {}, []
    for k, fm in enumerate(folds):
        rec = ReceiverModel().fit(R[~R.match_id.isin(fm)])
        rec_folds[k] = rec
        te_r = R.match_id.isin(fm).values
        R.loc[te_r, "p_oof"] = rec.predict(R[te_r], pd.factorize(R.frame_uid[te_r])[0])
        eng = NextThreatEngine(xt, xg, rec, None, None)
        for m in fm:
            agg = eng.pass_layer(ms[m], carriers[m])
            A_parts.append(action_rows(ms[m], carriers[m], agg, xt, xg))
        log(f"receiver fold {k}: T = {rec.temperature:.3f}")
    A = pd.concat(A_parts, ignore_index=True).sort_values(["match_id", "frame"]).reset_index(drop=True)
    receiver = ReceiverModel().fit(R)
    action = ActionModel().fit(A)
    K = A[(A.act == "keep") & A.dx.notna() & A.dy_out.notna()].reset_index(drop=True)
    carry = CarryModel().fit(K)
    engine = NextThreatEngine(xt, xg, receiver, action, carry)
    log(f"full fit: receiver T = {receiver.temperature:.3f}, action T = {action.temperature:.3f}")
    oof = []
    for k, fm in enumerate(folds):
        tr = A[~A.match_id.isin(fm)]
        act_k = ActionModel().fit(tr)
        car_k = CarryModel().fit(tr[(tr.act == "keep") & tr.dx.notna() & tr.dy_out.notna()])
        ek = NextThreatEngine(xt, xg, rec_folds[k], act_k, car_k)
        te = A[A.match_id.isin(fm)]
        PA = act_k.predict(te)
        sub = te[["match_id", "frame", "tp", "act", "tte", "ent", "xg_here"]].copy()
        sub["p_keep"], sub["p_pass"], sub["p_shot"], sub["p_other"] = PA.T
        cp = car_k.predict(te)
        (sub["carry_dx"], sub["carry_sd_dx"]), (sub["carry_dy_out"], sub["carry_sd_dy"]) = cp["dx"], cp["dy_out"]
        Vs = []
        for m, g in te.groupby("match_id", sort=False):
            V, _ = ek.carry_value(g, ms[m].pitch_length, ms[m].pitch_width); Vs.append(pd.Series(V, index=g.index))
        sub["v_carry"] = pd.concat(Vs)
        sub["ent_full"] = sub.p_keep * sub.v_carry + sub.p_pass * sub.ent + sub.p_shot * sub.xg_here
        oof.append(sub)
        log(f"oof fold {k}: action T = {act_k.temperature:.3f}")
    oof = pd.concat(oof, ignore_index=True)
    engine, oof = fit_stage2(engine, oof, A, ms, pp, carriers, folds, log)
    return SimpleNamespace(engine=engine, oof=oof, A=A, R=R, ms=ms, pp=pp, carriers=carriers, folds=folds)


def fit_stage2(engine, oof, A, ms, pp, carriers, folds=FOLDS, log=print):
    """Shot calibration and the danger layer, both fitted on out-of-match engine scores."""
    oof = oof.sort_values(["match_id", "frame"]).reset_index(drop=True)
    y_shot = (oof.act == "shot").values.astype(int)
    P = oof[["p_keep", "p_pass", "p_shot", "p_other"]].values
    Pc = np.zeros_like(P)
    fold_of = oof.match_id.map({m: k for k, fm in enumerate(folds) for m in fm}).values
    for k in range(len(folds)):
        te = fold_of == k
        Pc[te] = ShotCalibrator().fit(P[~te, 2], y_shot[~te])(P[te])
    engine.shot_calibrator = ShotCalibrator().fit(P[:, 2], y_shot)
    oof["p_keep_c"], oof["p_pass_c"], oof["p_shot_c"], oof["p_other_c"] = Pc.T
    oof["ent_full_raw"] = oof.ent_full
    oof["ent_full"] = oof.p_keep_c * oof.v_carry + oof.p_pass_c * oof.ent + oof.p_shot_c * oof.xg_here
    oof = oof.merge(A[["match_id", "frame", "max_mate_xt", "mean_mate_xt"]], on=["match_id", "frame"], how="left")
    parts = []
    for m, g in oof.groupby("match_id", sort=False):
        C = carriers[m].set_index("frame").loc[g.frame.values, ["carrier_slot", "sign"]].reset_index()
        g = g.reset_index(drop=True)
        g["carrier_slot"] = C.carrier_slot.values; g["sign"] = C.sign.values
        bf = ball_features(ms[m], g.frame.values.astype(int), g.carrier_slot.values.astype(int), g.sign.values.astype(float), engine.xt)
        tg = danger_targets(ms[m], g, pp)
        parts.append(pd.concat([g, bf, tg], axis=1))
    oof = trend_features(pd.concat(parts, ignore_index=True))
    D = oof[~oof.ball_in_box].reset_index(drop=True)
    pd_ = np.full(len(D), np.nan); pd0 = np.full(len(D), np.nan)
    fold_d = D.match_id.map({m: k for k, fm in enumerate(folds) for m in fm}).values
    no_trend = [c for c in DANGER_FEATURES if c not in TREND_FEATURES]
    for k in range(len(folds)):
        te = fold_d == k
        pd_[te] = DangerModel().fit(D[~te]).predict_raw(D[te])
        m0 = DangerModel(); m0.features = no_trend
        pd0[te] = m0.fit(D[~te]).predict_raw(D[te])
    # isotonic calibration, cross-fitted: fold k is mapped by a calibrator fitted on the other folds
    y = D.danger5.values
    pdc = np.zeros(len(D)); pd0c = np.zeros(len(D))
    for k in range(len(folds)):
        te = fold_d == k
        pdc[te] = DangerModel().calibrate(pd_[~te], y[~te]).predict_calibrated(pd_[te])
        pd0c[te] = DangerModel().calibrate(pd0[~te], y[~te]).predict_calibrated(pd0[te])
    D["p_danger5_raw"] = pd_; D["p_danger5"] = pdc; D["p_danger5_no_trend"] = pd0c
    engine.danger = DangerModel().fit(D).calibrate(pd_, y)
    log(f"danger layer: {len(D)} frames, base rate {D.danger5.mean():.3f}")
    oof = oof.merge(D[["match_id", "frame", "p_danger5", "p_danger5_raw", "p_danger5_no_trend"]], on=["match_id", "frame"], how="left")
    return engine, oof
