"""Out-of-match evidence for every claim in the README.

    python -m expected_next_threat.validate [output_folder]

Fits the engine fold by fold (five folds of four matches; each match is scored only by models that
never saw it), then reports:
  1. calibration: receiver, action (with P(shot) recalibration), carry, danger layer
  2. danger in the next 5 s beyond naive tracking (AUC, match-bootstrap 95% CIs)
  3. head to head with SkillCorner's live-usable outputs (possession EPV / xShot / xLoss, causal option menu)
Writes validation.json and evidence.png. Takes about 30 minutes on two cores. All results are
associations measured on matches the models did not see.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score

from .danger import DANGER_PARAMS, TREND_FEATURES
from .data import FPS
from .models import CLASSES, ActionModel
from .training import fit_engine
from .values import xt_at

NAIVE = ["bx", "by", "aby", "xt_ball", "max_mate_xt", "mean_mate_xt"]
SC_VALUES = ["sc_epv", "sc_xshot", "sc_xloss"]
SC_MENU = ["sc_opt_n", "sc_opt_max_xthreat", "sc_opt_sum_xthreat", "sc_opt_ev", "sc_opt_ndanger"]
LOCATION = ["cx", "acy", "dgoal", "ang_goal"]
B = 500
rng = np.random.default_rng(7)


def r4(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


def ece(p, y, bins=10, w=None):
    w = np.ones(len(p)) if w is None else w
    b = np.minimum((np.asarray(p) * bins).astype(int), bins - 1)
    n = np.bincount(b, weights=w, minlength=bins); s = np.bincount(b, weights=w * p, minlength=bins)
    o = np.bincount(b, weights=w * y, minlength=bins); nz = n > 0
    return float(np.abs(s[nz] - o[nz]).sum() / w.sum())


def reliability(p, y, bins=10):
    b = np.minimum((np.asarray(p) * bins).astype(int), bins - 1)
    out = []
    for k in range(bins):
        m = b == k
        if m.sum() >= 30:
            out.append(dict(mean_p=r4(p[m].mean()), observed=r4(y[m].mean()), n=int(m.sum())))
    return out


def cv_lgb(D, feats, target, folds):
    pr = np.zeros(len(D)); w = 1.0 / D.groupby("tp").tp.transform("size").values
    for fm in folds:
        te = D.match_id.isin(fm).values
        m = lgb.LGBMClassifier(**DANGER_PARAMS).fit(D.loc[~te, feats], D.loc[~te, target], sample_weight=w[~te])
        pr[te] = m.predict_proba(D.loc[te, feats])[:, 1]
    return pr


def boot_gain(D, pa, pb, target):
    """AUC(pb) - AUC(pa) with a match bootstrap (possession-weighted)."""
    y = D[target].values; w = 1.0 / D.groupby("tp").tp.transform("size").values
    gain = roc_auc_score(y, pb, sample_weight=w) - roc_auc_score(y, pa, sample_weight=w)
    idx = {m: np.where(D.match_id.values == m)[0] for m in D.match_id.unique()}
    ms = list(idx); ds = []
    for _ in range(B):
        ii = np.concatenate([idx[m] for m in rng.choice(ms, len(ms))])
        ds.append(roc_auc_score(y[ii], pb[ii], sample_weight=w[ii]) - roc_auc_score(y[ii], pa[ii], sample_weight=w[ii]))
    return dict(base=r4(roc_auc_score(y, pa, sample_weight=w)), plus=r4(roc_auc_score(y, pb, sample_weight=w)),
                gain=r4(gain), ci95=[r4(np.percentile(ds, 2.5)), r4(np.percentile(ds, 97.5))])


def receiver_section(R):
    R = R[R.p_oof.notna()].copy()
    g = pd.factorize(R.frame_uid)[0]; R["g"] = g
    R["rank"] = R.groupby("g").p_oof.rank(ascending=False, method="first")
    hit = R[R.y == 1]
    w = hit.w.values
    top = R[R["rank"] == 1]
    n_cand = R.groupby("g").size()
    return dict(frames=int(R.g.nunique()), passes=int(R.pass_uid.nunique()), mean_candidates=r4(n_cand.mean()),
                chance_top1=r4((1 / n_cand).mean()),
                top1=r4(np.average(hit["rank"].values == 1, weights=w)), top3=r4(np.average(hit["rank"].values <= 3, weights=w)),
                log_loss=r4(-np.average(np.log(np.clip(hit.p_oof.values, 1e-12, 1)), weights=w)),
                ece_candidate=r4(ece(R.p_oof.values, R.y.values, w=R.w.values)),
                ece_top1=r4(ece(top.p_oof.values, top.y.values.astype(float), w=top.w.values)),
                reliability_candidate=reliability(R.p_oof.values, R.y.values))


def action_section(fit):
    A, oof, folds = fit.A, fit.oof, fit.folds
    y = A.set_index(["match_id", "frame"]).y.reindex(pd.MultiIndex.from_frame(oof[["match_id", "frame"]])).values
    P = oof[["p_keep_c", "p_pass_c", "p_shot_c", "p_other_c"]].values
    Praw = oof[["p_keep", "p_pass", "p_shot", "p_other"]].values
    PL = np.zeros((len(A), 4))
    for fm in folds:
        te = A.match_id.isin(fm).values
        m = ActionModel(); m.features = LOCATION
        PL[te] = m.fit(A[~te]).predict(A[te])
    yl = A.y.values
    ll = lambda P, y: float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1)).mean())
    out = dict(frames=int(len(oof)), class_share={c: r4((y == i).mean()) for i, c in enumerate(CLASSES)},
               log_loss=r4(ll(P, y)), log_loss_location_only=r4(ll(PL, yl)),
               accuracy=r4((P.argmax(1) == y).mean()), accuracy_location_only=r4((PL.argmax(1) == yl).mean()),
               ece_by_class={c: r4(ece(P[:, i], (y == i).astype(float))) for i, c in enumerate(CLASSES)},
               shot_ece_before_recalibration=r4(ece(Praw[:, 2], (y == 2).astype(float), bins=20)),
               shot_ece_after_recalibration=r4(ece(P[:, 2], (y == 2).astype(float), bins=20)))
    return out


def carry_section(fit):
    oof, A = fit.oof, fit.A
    K = oof.merge(A[["match_id", "frame", "dx", "dy_out", "vx1", "vy1_out", "cx", "cy"]], on=["match_id", "frame"])
    K = K[(K.act == "keep") & K.dx.notna() & K.dy_out.notna()]
    cvx, cvy = K.vx1.fillna(0), K.vy1_out.fillna(0)
    L = K.match_id.map({m: M.pitch_length for m, M in fit.ms.items()}).values
    W = K.match_id.map({m: M.pitch_width for m, M in fit.ms.items()}).values
    sg = np.sign(K.cy.values + 1e-6)
    real = xt_at(fit.engine.xt, np.clip(K.cx + K.dx, -L / 2, L / 2), np.clip(K.cy + K.dy_out * sg, -W / 2, W / 2), L, W)
    cv = xt_at(fit.engine.xt, np.clip(K.cx + cvx, -L / 2, L / 2), np.clip(K.cy + cvy * sg, -W / 2, W / 2), L, W)
    return dict(frames=int(len(K)),
                r2_forward=r4(r2_score(K.dx, K.carry_dx)), r2_forward_constant_velocity=r4(r2_score(K.dx, cvx)),
                r2_outward=r4(r2_score(K.dy_out, K.carry_dy_out)), r2_outward_constant_velocity=r4(r2_score(K.dy_out, cvy)),
                within_1sd_forward=r4((np.abs(K.dx - K.carry_dx) <= K.carry_sd_dx).mean()),
                within_1sd_outward=r4((np.abs(K.dy_out - K.carry_dy_out) <= K.carry_sd_dy).mean()),
                carry_value_r2_v_realised_xt=r4(r2_score(real, K.v_carry)),
                constant_velocity_r2_v_realised_xt=r4(r2_score(real, cv)))


def skillcorner_features(fit, D):
    parts = []
    for m, g in D.groupby("match_id", sort=False):
        ev = fit.ms[m].events
        C = fit.carriers[m].set_index("frame").loc[g.frame.values, ["event_id", "frame_end"]].reset_index()
        g = g.reset_index(drop=True); g["event_id"] = C.event_id.values; g["pp_frame_end"] = C.frame_end.values
        pp = ev[ev.event_type == "player_possession"][["event_id", "possession_epv_for_start", "xshot_player_possession_start",
                                                       "xloss_player_possession_start"]]
        g = g.merge(pp.rename(columns={"possession_epv_for_start": "sc_epv", "xshot_player_possession_start": "sc_xshot",
                                       "xloss_player_possession_start": "sc_xloss"}), on="event_id", how="left")
        po = ev[ev.event_type == "passing_option"][["associated_player_possession_event_id", "frame_start", "frame_end",
                                                    "peak_passing_option_frame", "xthreat", "xpass_completion", "dangerous"]]
        po = po.rename(columns={"associated_player_possession_event_id": "event_id", "frame_start": "o0", "frame_end": "o1"}).dropna(subset=["event_id"])
        po["ev"] = po.xthreat * po.xpass_completion
        po["dangerous"] = po.dangerous.astype(str).str.lower().eq("true").astype(float)
        J = g[["event_id", "frame"]].reset_index().merge(po, on="event_id")
        J = J[(J.o0 <= J.frame) & (J.frame <= J.o1) & (J.peak_passing_option_frame <= J.frame)]   # causal: already peaked
        agg = J.groupby("index").agg(sc_opt_n=("xthreat", "size"), sc_opt_max_xthreat=("xthreat", "max"),
                                     sc_opt_sum_xthreat=("xthreat", "sum"), sc_opt_ev=("ev", "sum"), sc_opt_ndanger=("dangerous", "sum"))
        g = g.join(agg)
        parts.append(g)
    D = pd.concat(parts, ignore_index=True)
    for c in SC_MENU:
        D[c] = D[c].fillna(0.0)
    return D


def danger_section(fit):
    D = fit.oof[fit.oof.p_danger5.notna()].reset_index(drop=True)
    y = D.danger5.values.astype(float); w = 1.0 / D.groupby("tp").tp.transform("size").values
    out = dict(frames=int(len(D)), possessions=int(D.tp.nunique()), base_rate=r4(np.average(y, weights=w)),
               auc=r4(roc_auc_score(y, D.p_danger5, sample_weight=w)), ece=r4(ece(D.p_danger5.values, y)),
               ece_before_calibration=r4(ece(D.p_danger5_raw.values, y)), reliability=reliability(D.p_danger5.values, y))
    folds = fit.folds
    p_naive = cv_lgb(D, NAIVE, "danger5", folds)
    p_ent = cv_lgb(D, NAIVE + ["ent_full"], "danger5", folds)
    p_full = cv_lgb(D, NAIVE + ["ent_full"] + TREND_FEATURES, "danger5", folds)
    out["engine_over_naive_tracking"] = boot_gain(D, p_naive, p_ent, "danger5")
    out["trend_over_engine"] = boot_gain(D, p_ent, p_full, "danger5")
    # head to head with SkillCorner, frames more than 1 s before the player possession ends
    S = skillcorner_features(fit, D)
    S = S[(S.pp_frame_end - S.frame) > FPS].reset_index(drop=True)
    sc = {}
    for name, feats in (("possession_values", SC_VALUES), ("causal_option_menu", SC_MENU), ("both", SC_VALUES + SC_MENU)):
        p_sc = cv_lgb(S, NAIVE + feats, "danger5", folds)
        p_us = cv_lgb(S, NAIVE + ["ent_full"], "danger5", folds)
        p_both = cv_lgb(S, NAIVE + feats + ["ent_full"], "danger5", folds)
        sc[name] = dict(head_to_head_engine_minus_skillcorner=boot_gain(S, p_sc, p_us, "danger5"),
                        engine_beyond_skillcorner=boot_gain(S, p_sc, p_both, "danger5"))
    out["skillcorner"] = dict(frames=int(len(S)), share_with_causal_option=r4((S.sc_opt_n > 0).mean()), **sc)
    return out


def evidence_figure(rep, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw=dict(width_ratios=[1, 1.1]))
    a = ax[0]
    a.plot([0, 1], [0, 1], color="#c9cdc9", lw=1)
    for key, lab, col in (("receiver", "Next receiver (per team-mate)", "#c2410c"), ("danger", "Danger in the next 5 s", "#0f6b5c")):
        rel = [r for r in rep[key]["reliability_candidate" if key == "receiver" else "reliability"] if r["n"] >= 100]
        a.plot([r["mean_p"] for r in rel], [r["observed"] for r in rel], "o-", color=col, lw=2, ms=4, label=lab)
    a.set_xlabel("predicted probability"); a.set_ylabel("observed frequency"); a.set_xlim(0, 1); a.set_ylim(0, 1)
    a.set_title("Calibration on unseen matches", loc="left", fontsize=11); a.legend(frameon=False, fontsize=9, loc="upper left")
    b = ax[1]
    d = rep["danger"]
    rows = [("over ball position\n+ team-mate xT", d["engine_over_naive_tracking"]),
            ("v SkillCorner\npossession EPV, xShot, xLoss", d["skillcorner"]["possession_values"]["head_to_head_engine_minus_skillcorner"]),
            ("v SkillCorner\npassing options (live)", d["skillcorner"]["causal_option_menu"]["head_to_head_engine_minus_skillcorner"]),
            ("v both SkillCorner sets", d["skillcorner"]["both"]["head_to_head_engine_minus_skillcorner"])]
    yv = np.arange(len(rows))[::-1]
    for yi, (lab, r) in zip(yv, rows):
        b.errorbar(r["gain"], yi, xerr=[[r["gain"] - r["ci95"][0]], [r["ci95"][1] - r["gain"]]], fmt="o", color="#c2410c", capsize=3, lw=1.5)
    b.axvline(0, color="#9aa3a0", lw=1)
    b.set_yticks(yv); b.set_yticklabels([r[0] for r in rows], fontsize=9)
    b.set_xlabel("AUC gain (95% CI, match bootstrap)")
    b.set_title("Danger in the next 5 s: ENT's added AUC", loc="left", fontsize=11)
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def main(out_dir="validation"):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    fit = fit_engine(log=lambda s: print(s, flush=True))
    rep = dict(note="All numbers are out of match: each match is scored by models that never saw it. Associations only.",
               folds=[list(map(int, f)) for f in fit.folds])
    rep["receiver"] = receiver_section(fit.R); print("receiver done", flush=True)
    rep["action"] = action_section(fit); print("action done", flush=True)
    rep["carry"] = carry_section(fit); print("carry done", flush=True)
    rep["danger"] = danger_section(fit); print("danger done", flush=True)
    (out / "validation.json").write_text(json.dumps(rep, indent=1))
    evidence_figure(rep, out / "evidence.png")
    fit.oof.to_parquet(out / "oof_frames.parquet")
    print(json.dumps({k: v for k, v in rep.items() if k != "folds"}, indent=1))
    return rep


if __name__ == "__main__":
    main(*sys.argv[1:])
