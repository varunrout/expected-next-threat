"""Train the pretrained engine on all 20 matches and write out-of-match scores for validation.

    python scripts/train.py            # writes src/expected_next_threat/pretrained/ and oof_frames.parquet
"""
import sys
from pathlib import Path
from expected_next_threat.engine import PRETRAINED
from expected_next_threat.training import fit_engine

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("build")
out.mkdir(exist_ok=True, parents=True)
engine, oof, A, R = fit_engine( log=lambda s: print(s, flush=True))
engine.save(PRETRAINED)
oof.to_parquet(out / "oof_frames.parquet")
A.to_parquet(out / "action_rows.parquet")
print("receiver rows", len(R), "action rows", len(A))
