"""Team possessions and on-ball (carrier) frames."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import FPS, Match

STEP = 2  # carrier frames every 0.2 s


def stitch_possessions(events: pd.DataFrame):
    """Group player possessions into team possessions.

    A new team possession starts when the team changes, or a game interruption sits before this
    player possession or after the previous one. Works on one match or many (ids are global).
    Returns (pp, tp): player possessions with a `tp` column, and one row per team possession.
    """
    pp = events[events.event_type == "player_possession"].sort_values(["match_id", "period", "frame_start"]).copy()
    g = pp.groupby(["match_id", "period"])
    prev_team = g.team_id.shift(1)
    prev_int = g.game_interruption_after.shift(1)
    new = (pp.team_id != prev_team) | pp.game_interruption_before.notna() | prev_int.notna() | prev_team.isna()
    pp["tp"] = new.cumsum()
    pp["danger_ev"] = (pp.end_type == "shot") | pp.penalty_area_start.fillna(False).astype(bool) \
        | pp.penalty_area_end.fillna(False).astype(bool)
    tp = pp.groupby("tp").agg(match_id=("match_id", "first"), period=("period", "first"), team_id=("team_id", "first"),
                              side=("attacking_side", "first"), f0=("frame_start", "min"), f1=("frame_end", "max"),
                              n_pp=("event_id", "size"), danger=("danger_ev", "max"),
                              phase=("team_in_possession_phase_type", "first"),
                              setpiece=("game_interruption_before", lambda s: s.notna().iloc[0]))
    return pp, tp


def carrier_frames(match: Match, pp: pd.DataFrame, drop_untracked: bool = True) -> pd.DataFrame:
    """Every 0.2 s frame where a tracked player holds the ball (one carrier per frame).

    Columns: frame, carrier_slot, tp, team_id, sign (+1 if the team attacks left to right),
    t_on_ball (s since the player possession started), frame_start, frame_end, end_type.
    """
    ppm = pp[pp.match_id == match.match_id]
    rows = []
    for r in ppm.itertuples():
        if r.player_id not in match.slot:
            continue
        fs = np.arange(int(r.frame_start), int(r.frame_end) + 1)
        fs = fs[(fs % STEP == 0) & (fs >= FPS) & (fs < match.nf)]
        if len(fs) == 0:
            continue
        rows.append(pd.DataFrame(dict(frame=fs, carrier_slot=match.slot[r.player_id], tp=r.tp, team_id=r.team_id,
                                      sign=1.0 if r.attacking_side == "left_to_right" else -1.0,
                                      t_on_ball=(fs - int(r.frame_start)) / FPS, frame_start=int(r.frame_start),
                                      frame_end=int(r.frame_end), end_type=r.end_type)))
    if not rows:
        return pd.DataFrame(columns=["frame", "carrier_slot", "tp", "team_id", "sign", "t_on_ball", "frame_start", "frame_end", "end_type"])
    C = pd.concat(rows, ignore_index=True)
    # one carrier per frame: keep the first player possession that covers it (as in the analysis)
    C = C.sort_values("frame", kind="stable").drop_duplicates("frame", keep="first").reset_index(drop=True)
    if drop_untracked:  # carrier has no position in the tracking (rare: about 0.2% of frames)
        C = C[~np.isnan(match.X[C.frame.values, C.carrier_slot.values])].reset_index(drop=True)
    return C
