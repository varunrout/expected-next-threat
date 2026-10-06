"""expected-next-threat: where the ball and the danger go next, every 0.2 s, on SkillCorner tracking data."""
from .data import MATCH_IDS, load_match
from .engine import NextThreatEngine

__all__ = ["NextThreatEngine", "load_match", "MATCH_IDS"]
__version__ = "0.1.0.dev0"
