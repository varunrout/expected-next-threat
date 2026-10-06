# expected-next-threat

Work in progress for the PySport x SkillCorner Analytics Cup 2.0.

Every 0.2 s that a player holds the ball, the engine estimates what happens in the next second (keep, pass, shoot or lose it), who would receive a pass and where a carry would end, and values each outcome with expected threat (xT) or expected goals (xG). The sum is the **expected next threat (ENT)**.

```python
from expected_next_threat import NextThreatEngine, load_match

engine = NextThreatEngine.pretrained()
match = load_match(1996435)          # SkillCorner open data; downloaded and cached on first use
live = engine.predict_match(match)   # one row per 0.2 s on-ball frame
```

Data: SkillCorner open data (A-League 2024/25), pinned to commit `4340d27`.
