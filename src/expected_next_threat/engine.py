"""NextThreatEngine: expected next threat (ENT) at every on-ball frame.

ENT = P(keep) x E[xT where a 1 s carry ends] + P(pass) x sum_j P(receiver = j) xT(where j will be)
      + P(shot) x xG(here) + P(other) x 0
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import FPS, Match
from .features import candidate_features, carrier_features, receiver_targets
from .models import ActionModel, CarryModel, ReceiverModel, load_booster, save_booster
from .possessions import carrier_frames, stitch_possessions
from .values import XG, xt_at

PRETRAINED = Path(__file__).parent / "pretrained"
_GH_NODES, _GH_W = np.polynomial.hermite_e.hermegauss(5)
_GH_W = _GH_W / _GH_W.sum()


class NextThreatEngine:
    def __init__(self, xt: np.ndarray, xg: XG, receiver: ReceiverModel, action: ActionModel, carry: CarryModel):
        self.xt, self.xg, self.receiver, self.action, self.carry = xt, xg, receiver, action, carry

    # ------------------------------------------------------------------ layers
    def pass_layer(self, M: Match, C: pd.DataFrame, return_candidates: bool = False):
        """Receiver probabilities and pass value for carrier frames C (frame, carrier_slot, team_id, sign, t_on_ball)."""
        L, W = M.pitch_length, M.pitch_width
        f = C.frame.values.astype(int); cs = C.carrier_slot.values.astype(int)
        tm = C.team_id.values; sg = C.sign.values.astype(float); n = len(C)
        Cf = candidate_features(M, f, cs, tm, sg, L, W)
        g = Cf.row.values
        Cf["t_on_ball"] = C.t_on_ball.values[g]
        P = self.receiver.predict(Cf, g) if len(Cf) else np.zeros(0)
        tx, ty = receiver_targets(M, f[g], Cf.slot.values, cs[g], sg[g], L, W)
        xtt = xt_at(self.xt, tx, ty, L, W)
        agg = pd.DataFrame(dict(row=g, e=P * xtt, x=xtt, p=P)).groupby("row").agg(
            ent=("e", "sum"), max_mate_xt=("x", "max"), mean_mate_xt=("x", "mean"), pmax=("p", "max")).reindex(np.arange(n))
        if not return_candidates:
            return agg
        cand = pd.DataFrame(dict(frame=f[g], slot=Cf.slot.values, player_id=M.pids[Cf.slot.values], p_receiver=P,
                                 xt_target=xtt, x=Cf.xj.values, y=Cf.yj.values, target_x=tx, target_y=ty))
        return agg, cand

    def carry_value(self, F: pd.DataFrame, L: float, W: float):
        cp = self.carry.predict(F)
        (mdx, sdx), (mdy, sdy) = cp["dx"], cp["dy_out"]
        cx, cy = F.cx.values, F.cy.values; sgn = np.sign(cy + 1e-6)
        V = np.zeros(len(F))
        for a, wa in zip(_GH_NODES, _GH_W):
            for b, wb in zip(_GH_NODES, _GH_W):
                ex = np.clip(cx + mdx + a * sdx, -L / 2, L / 2); ey = np.clip(cy + (mdy + b * sdy) * sgn, -W / 2, W / 2)
                V += wa * wb * xt_at(self.xt, ex, ey, L, W)
        return V, dict(carry_dx=mdx, carry_dy_out=mdy, carry_sd_dx=sdx, carry_sd_dy=sdy)

    def predict_frames(self, M: Match, C: pd.DataFrame, pass_layer: pd.DataFrame | None = None) -> pd.DataFrame:
        """Full engine on carrier frames C. Returns one row per frame."""
        L, W = M.pitch_length, M.pitch_width
        f = C.frame.values.astype(int); cs = C.carrier_slot.values.astype(int)
        tm = C.team_id.values; sg = C.sign.values.astype(float)
        agg = self.pass_layer(M, C) if pass_layer is None else pass_layer
        F = carrier_features(M, f, cs, tm, sg, C.t_on_ball.values, L, W, self.xt, self.xg)
        for c in ("pmax", "max_mate_xt", "mean_mate_xt"):
            F[c] = agg[c].values
        F["ent"] = agg.ent.fillna(0).values
        PA = self.action.predict(F)
        V, cv = self.carry_value(F, L, W)
        out = C.reset_index(drop=True).copy()
        out["p_keep"], out["p_pass"], out["p_shot"], out["p_other"] = PA.T
        out["ent_pass"] = F.ent.values
        out["v_carry"] = V
        out["xg_here"] = F.xg_here.values; out["xt_here"] = F.xt_here.values
        out["ent_carry_part"] = out.p_keep * V
        out["ent_pass_part"] = out.p_pass * out.ent_pass
        out["ent_shot_part"] = out.p_shot * out.xg_here
        out["ent_full"] = out.ent_carry_part + out.ent_pass_part + out.ent_shot_part
        for k, v in cv.items():
            out[k] = v
        out["cx"], out["cy"] = F.cx.values, F.cy.values
        out["pmax"], out["max_mate_xt"], out["mean_mate_xt"] = F.pmax.values, F.max_mate_xt.values, F.mean_mate_xt.values
        return out

    def predict_match(self, match: Match) -> pd.DataFrame:
        """ENT for every 0.2 s on-ball frame of a match."""
        pp, _ = stitch_possessions(match.events)
        C = carrier_frames(match, pp)
        out = self.predict_frames(match, C)
        out.insert(0, "match_id", match.match_id)
        out["t"] = out.frame / FPS
        return out

    # ------------------------------------------------------------------ persistence
    def save(self, folder: Path) -> None:
        folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
        save_booster(self.receiver.booster, folder / "receiver.txt")
        save_booster(self.action.booster, folder / "action.txt")
        for tgt, (mu, sd) in self.carry.boosters.items():
            save_booster(mu, folder / f"carry_{tgt}_mean.txt"); save_booster(sd, folder / f"carry_{tgt}_spread.txt")
        meta = dict(xt=self.xt.tolist(), xg=self.xg.to_dict(), receiver_temperature=self.receiver.temperature,
                    action_temperature=self.action.temperature)
        (folder / "meta.json").write_text(json.dumps(meta, indent=1))

    @classmethod
    def load(cls, folder: Path) -> "NextThreatEngine":
        folder = Path(folder)
        meta = json.loads((folder / "meta.json").read_text())
        carry = CarryModel({t: (load_booster(folder / f"carry_{t}_mean.txt"), load_booster(folder / f"carry_{t}_spread.txt"))
                            for t in CarryModel.targets})
        return cls(np.asarray(meta["xt"]), XG(**meta["xg"]),
                   ReceiverModel(load_booster(folder / "receiver.txt"), meta["receiver_temperature"]),
                   ActionModel(load_booster(folder / "action.txt"), meta["action_temperature"]), carry)

    @classmethod
    def pretrained(cls) -> "NextThreatEngine":
        """Models trained on all 20 open-data matches (so predictions on them are in-sample)."""
        return cls.load(PRETRAINED)
