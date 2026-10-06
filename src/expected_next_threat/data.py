"""Load SkillCorner open data (A-League 2024/25) straight from the public repository.

Files are downloaded once per match into a local cache. The source commit is pinned so every
number the package reports can be reproduced on the same data.

    match = load_match(1996435)
    match.events      # dynamic events (attack-normalised x/y)
    match.X[f, k]     # raw tracking x for frame f, player slot k (NaN when absent)
"""
from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

FPS = 10
SOURCE_REPO = "SkillCorner/opendata"
SOURCE_COMMIT = "4340d274572876239c154c90bc507a9b3250a656"
MATCH_IDS = (1874553, 1886347, 1899585, 1925299, 1927964, 1953632, 1959846, 1986691, 1996435, 1996436,
             2006229, 2006363, 2007448, 2007721, 2010085, 2011166, 2013725, 2015213, 2016236, 2017461)
_RAW = f"https://raw.githubusercontent.com/{SOURCE_REPO}/{SOURCE_COMMIT}/data/matches"
_LFS = f"https://media.githubusercontent.com/media/{SOURCE_REPO}/{SOURCE_COMMIT}/data/matches"


def cache_dir() -> Path:
    """Cache folder: $EXPECTED_NEXT_THREAT_CACHE, else ~/.cache/expected-next-threat."""
    p = Path(os.environ.get("EXPECTED_NEXT_THREAT_CACHE", Path.home() / ".cache" / "expected-next-threat"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as fh:
        while chunk := r.read(1 << 20):
            fh.write(chunk)
    tmp.replace(dest)


def fetch_match(match_id: int, quiet: bool = False) -> Path:
    """Download the four raw files of one match (about 95 MB) if they are not cached yet."""
    folder = cache_dir() / "raw" / str(match_id)
    folder.mkdir(parents=True, exist_ok=True)
    files = {f"{match_id}_dynamic_events.csv": _RAW, f"{match_id}_match.json": _RAW,
             f"{match_id}_phases_of_play.csv": _RAW, f"{match_id}_tracking_extrapolated.jsonl": _LFS}
    for name, base in files.items():
        dest = folder / name
        if not dest.exists():
            if not quiet:
                print(f"downloading {name}")
            _download(f"{base}/{match_id}/{name}", dest)
    return folder


@dataclass
class Match:
    """One match: events, metadata and dense tracking arrays.

    Arrays are indexed [frame, slot]. Coordinates are raw pitch metres (centre origin), as in the
    SkillCorner tracking files; use `sign` per possession to make the team in possession attack +x.
    """
    match_id: int
    pitch_length: float
    pitch_width: float
    events: pd.DataFrame
    players: pd.DataFrame
    pids: np.ndarray
    team: np.ndarray
    gk: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    DET: np.ndarray
    BX: np.ndarray
    BY: np.ndarray
    BDET: np.ndarray
    slot: dict = field(default_factory=dict)

    @property
    def nf(self) -> int:
        return self.X.shape[0]


def _parse_tracking(path: Path, pid_slot: dict):
    frames, ks, xs, ys, ds = [], [], [], [], []
    bf, bx, by, bd = [], [], [], []
    nf = 0
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            f = r["frame"]; nf = max(nf, f + 1)
            b = r.get("ball_data") or {}
            if b.get("x") is not None:
                bf.append(f); bx.append(b["x"]); by.append(b["y"]); bd.append(bool(b.get("is_detected")))
            for p in r.get("player_data") or ():
                k = pid_slot.get(p["player_id"])
                if k is None or p.get("x") is None:
                    continue
                frames.append(f); ks.append(k); xs.append(p["x"]); ys.append(p["y"]); ds.append(bool(p.get("is_detected")))
    return nf, (np.asarray(frames), np.asarray(ks), np.asarray(xs), np.asarray(ys), np.asarray(ds)), \
        (np.asarray(bf), np.asarray(bx), np.asarray(by), np.asarray(bd))


def load_match(match_id: int, quiet: bool = False) -> Match:
    """Load one match, downloading and caching it on first use. Parsed arrays are cached as .npz."""
    match_id = int(match_id)
    folder = fetch_match(match_id, quiet=quiet)
    meta = json.loads((folder / f"{match_id}_match.json").read_text())
    events = pd.read_csv(folder / f"{match_id}_dynamic_events.csv", low_memory=False)
    events["match_id"] = match_id
    pl = pd.DataFrame([dict(player_id=p["id"], team_id=p["team_id"], short_name=p.get("short_name"),
                            is_gk=(p.get("player_role") or {}).get("acronym") == "GK",
                            start_time=p.get("start_time")) for p in meta["players"]])
    pl = pl[pl.start_time.notna()].drop(columns="start_time").reset_index(drop=True)
    pids = np.sort(pl.player_id.unique())
    slot = {int(p): i for i, p in enumerate(pids)}
    pl = pl.set_index("player_id").loc[pids].reset_index()
    npz = folder / "arrays.npz"
    if npz.exists():
        a = np.load(npz)
        X, Y, DET, BX, BY, BDET = (a[k] for k in ("X", "Y", "DET", "BX", "BY", "BDET"))
    else:
        nf, (f, k, x, y, d), (bf, bx, by, bd) = _parse_tracking(folder / f"{match_id}_tracking_extrapolated.jsonl", slot)
        K = len(pids)
        X = np.full((nf, K), np.nan, np.float32); Y = X.copy(); DET = np.zeros((nf, K), bool)
        X[f, k] = x; Y[f, k] = y; DET[f, k] = d
        BX = np.full(nf, np.nan, np.float32); BY = BX.copy(); BDET = np.zeros(nf, bool)
        BX[bf] = bx; BY[bf] = by; BDET[bf] = bd
        np.savez_compressed(npz, X=X, Y=Y, DET=DET, BX=BX, BY=BY, BDET=BDET)
    return Match(match_id=match_id, pitch_length=float(meta["pitch_length"]), pitch_width=float(meta["pitch_width"]),
                 events=events, players=pl, pids=pids, team=pl.team_id.values, gk=pl.is_gk.values.astype(bool),
                 X=X, Y=Y, DET=DET, BX=BX, BY=BY, BDET=BDET, slot=slot)
