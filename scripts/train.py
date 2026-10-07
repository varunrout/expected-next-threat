"""Train the pretrained engine on all 20 matches.

    python scripts/train.py          # writes src/expected_next_threat/pretrained/

About 25 minutes on two cores. `python -m expected_next_threat.validate` runs the same fit and
reports the out-of-match evidence.
"""
from expected_next_threat.engine import PRETRAINED
from expected_next_threat.training import fit_engine

fit = fit_engine(log=lambda s: print(s, flush=True))
fit.engine.save(PRETRAINED)
print("saved", PRETRAINED)
