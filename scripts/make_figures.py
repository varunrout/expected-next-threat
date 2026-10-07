"""Rebuild the README figures and the video clip.

    python scripts/make_figures.py            # Figure 1 (needs only the pretrained engine)
    python -m expected_next_threat.validate docs/validation   # Figure 2 and validation.json (about 30 min)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from expected_next_threat import NextThreatEngine, load_match
from expected_next_threat.viz import animate_possession, plot_moment

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs")
out.mkdir(parents=True, exist_ok=True)
MATCH, POSSESSION = 1996435, 199          # Sydney FC's goal possession v Adelaide United (74:10)
engine = NextThreatEngine.pretrained()
match = load_match(MATCH)
live = engine.predict_match(match)
fig = plot_moment(engine, match, live, POSSESSION, title="Sydney FC v Adelaide United, 74:10: the possession that ends in a goal")
fig.savefig(out / "figure1_moment.png", dpi=150, bbox_inches="tight")
if "--video" in sys.argv:
    animate_possession(engine, match, live, POSSESSION, str(out / "goal_possession.mp4"), fps=5)
print("written to", out)
