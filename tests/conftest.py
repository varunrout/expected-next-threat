import numpy as np
import pandas as pd
import pytest

from expected_next_threat.data import Match


def make_match(seed=0, nf=200, L=105.0, W=68.0):
    """A synthetic match: 22 players moving smoothly, team 1 attacking left to right."""
    rng = np.random.default_rng(seed)
    K = 22
    team = np.array([1] * 11 + [2] * 11)
    gk = np.zeros(K, bool); gk[0] = gk[11] = True
    base_x = np.r_[-48, rng.uniform(-30, 30, 10), 48, rng.uniform(-30, 30, 10)]
    base_y = rng.uniform(-28, 28, K)
    t = np.arange(nf)[:, None] / 10
    X = (base_x + 2 * np.sin(t / 3 + np.arange(K))).astype(np.float32)
    Y = (base_y + 2 * np.cos(t / 4 + np.arange(K))).astype(np.float32)
    BX = X[:, 5].copy(); BY = Y[:, 5].copy()
    players = pd.DataFrame(dict(player_id=np.arange(100, 100 + K), team_id=team, short_name=[f"p{k}" for k in range(K)], is_gk=gk))
    return Match(match_id=1, pitch_length=L, pitch_width=W, events=pd.DataFrame(), players=players,
                 pids=players.player_id.values, team=team, gk=gk, X=X, Y=Y, DET=np.ones((nf, K), bool),
                 BX=BX, BY=BY, BDET=np.ones(nf, bool), slot={int(p): i for i, p in enumerate(players.player_id)})


def carriers(frames, slot=5, team=1, sign=1.0):
    frames = np.asarray(frames)
    return pd.DataFrame(dict(frame=frames, carrier_slot=slot, tp=1, team_id=team, sign=sign,
                             t_on_ball=(frames - frames[0]) / 10, frame_start=frames[0], frame_end=frames[-1] + 30, end_type="pass"))


@pytest.fixture
def match():
    return make_match()
