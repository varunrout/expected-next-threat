import numpy as np
import pytest

from expected_next_threat.engine import PRETRAINED, NextThreatEngine
from expected_next_threat.features import candidate_features, carrier_features
from expected_next_threat.values import NX, NY, XG, _cell, xt_at

from conftest import carriers, make_match


def test_cells_stay_on_grid():
    x = np.array([-60, -52.5, 0, 52.4, 60]); y = np.array([-40, -34, 0, 33.9, 40])
    c = _cell(x, y, 105, 68)
    assert c.min() >= 0 and c.max() < NX * NY
    assert _cell(52.4, 33.9, 105, 68) == NX * NY - 1


def test_xg_falls_with_distance():
    xg = XG(coef=[-0.12, 1.2], intercept=-0.5)
    near, far = xg(np.array([47.0]), np.array([0.0]), 105), xg(np.array([25.0]), np.array([0.0]), 105)
    assert near[0] > far[0] > 0


def test_candidates_are_outfield_teammates_without_carrier(match):
    f = np.array([50, 60]); cs = np.array([5, 5]); tm = np.array([1, 1]); s = np.array([1.0, 1.0])
    C = candidate_features(match, f, cs, tm, s, 105, 68)
    assert C.groupby("row").size().tolist() == [9, 9]          # 11 - goalkeeper - carrier
    assert not set(C.slot) & {0, 5}
    assert set(C.slot) <= set(range(11))


def test_direction_flip_is_invariant():
    a = make_match(seed=3)
    b = make_match(seed=3); b.X = -b.X; b.Y = -b.Y; b.BX = -b.BX; b.BY = -b.BY
    f = np.array([40, 80, 120]); cs = np.full(3, 5); tm = np.ones(3, int)
    ca = candidate_features(a, f, cs, tm, np.ones(3), 105, 68)
    cb = candidate_features(b, f, cs, tm, -np.ones(3), 105, 68)
    cols = [c for c in ca.columns if c not in ("row", "slot", "det_j")]
    np.testing.assert_allclose(ca[cols].values, cb[cols].values, rtol=1e-5, atol=1e-4)
    xt = np.linspace(0, 0.3, NX * NY); xg = XG(coef=[-0.12, 1.2], intercept=-0.5)
    fa = carrier_features(a, f, cs, tm, np.ones(3), np.zeros(3), 105, 68, xt, xg)
    fb = carrier_features(b, f, cs, tm, -np.ones(3), np.zeros(3), 105, 68, xt, xg)
    np.testing.assert_allclose(fa.drop(columns=["det_c", "det_d1"]).values, fb.drop(columns=["det_c", "det_d1"]).values, rtol=1e-5, atol=1e-4)


@pytest.mark.skipif(not (PRETRAINED / "meta.json").exists(), reason="pretrained models not built")
def test_engine_outputs_are_probabilities(match):
    eng = NextThreatEngine.pretrained()
    C = carriers(np.arange(20, 180, 2))
    out = eng.predict_frames(match, C)
    P = out[["p_keep", "p_pass", "p_shot", "p_other"]].values
    np.testing.assert_allclose(P.sum(1), 1, atol=1e-9)
    assert (P >= 0).all() and (out.ent_full >= 0).all()
    np.testing.assert_allclose(out.ent_full, out.ent_carry_part + out.ent_pass_part + out.ent_shot_part)
    agg, cand = eng.pass_layer(match, C, return_candidates=True)
    np.testing.assert_allclose(cand.groupby("frame").p_receiver.sum().values, 1, atol=1e-9)
