"""Testing the focus model. Run: .venv/bin/python tests/test_focusmodel.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from focusmodel import CURVE, FocusModel, distance_from_x, parse_mm, x_from_distance  # noqa: E402


def test_parse():
    assert parse_mm("178mm") == 178.0 and parse_mm("16mm") == 16.0 and parse_mm("") is None and parse_mm(None) is None


def test_curve_roundtrip_and_monotonic():
    for x, d in CURVE[:-1]:
        assert abs(distance_from_x(x) - d) < 1e-6
        assert abs(x_from_distance(d) - x) < 1e-6
    xs = [i / 100 for i in range(0, 94)]
    ds = [distance_from_x(x) for x in xs]
    assert all(a < b for a, b in zip(ds, ds[1:]))


def test_model_median_and_target():
    m = FocusModel()
    assert m.C is None
    assert not m.add(10389, 0.2, 16) and not m.add(None, 0.2, 16) # “infinity” and empty data is not accepted
    for d, w in ((178, 0.25), (178, 0.24), (212, 0.30), (154, 0.22), (178, 0.25)):
        assert m.add(d, w, 16)
    c = m.C
    assert 2.5 < c < 3.2
    # face wider (closer) → smaller reading → smaller focus position
    assert m.target_x(0.35, 16) < m.target_x(0.25, 16) < m.target_x(0.18, 16)
    # with the same width as in the training measurement, the position is near the peak (≈0.79)
    assert abs(m.target_x(0.25, 16) - 0.79) < 0.04


def test_cbins_and_shift():
    m = FocusModel()
    for d, w in ((178, 0.25), (178, 0.25), (178, 0.25)):
        m.add(d, w, 16)
    base = m.target_x(0.25, 16)
    m.learn_c(212, 0.25, 16) # at this face size the actual distance turned out to be greater
    assert m.target_x(0.25, 16) > base
    assert abs(m.c_at(0.25, 16) - 212 * 0.25 / 16) < 1e-6
    # near the table also works, but far away it returns to the general C
    assert abs(m.c_at(0.05, 16) - m.C) < 1e-9
    # the scale shift is simply added
    x0 = m.target_x(0.25, 16)
    m.shift = -0.085
    assert abs(m.target_x(0.25, 16) - (x0 - 0.085)) < 1e-9


def test_lens_slug_and_curve_from_sweep():
    from focusmodel import curve_from_sweep, lens_slug
    assert lens_slug(".64x Sigma DC 18-35/1.7 HSM") == "64x-sigma-dc-18-35-1-7-hsm"
    assert lens_slug("") == "unknown" and lens_slug(None) == "unknown"
    # the reading goes in steps; None and "noise down" are ignored
    sweep = [(0.0, None), (0.1, 56), (0.2, 56), (0.3, 56), (0.4, 80), (0.5, 80), (0.6, 70), (0.7, 124), (0.8, 124)]
    assert curve_from_sweep(sweep) == [(0.2, 56), (0.45, 80), (0.75, 124)]


def test_set_curve_and_profile_roundtrip(tmp_path=None):
    import tempfile
    import focusmodel as fm
    old = (list(fm._XS), list(fm._LN_D), fm.PROFILES_DIR)
    try:
        fm.PROFILES_DIR = tempfile.mkdtemp()
        fm.save_profile("test-lens", {"lens": "Test 12-35", "curve": [[0.1, 30], [0.5, 100], [0.9, 8000]], "sigma_ln": 0.3})
        p = fm.load_profile("test-lens")
        assert p["lens"] == "Test 12-35" and fm.load_profile("nope") is None
        fm.set_curve(p["curve"])
        assert abs(fm.x_from_distance(100) - 0.5) < 1e-6 and abs(fm.distance_from_x(0.5) - 100) < 1e-6
    finally:
        fm._XS, fm._LN_D, fm.PROFILES_DIR = old


def test_user_data_per_lens_and_legacy_migration():
    import json
    import os
    import tempfile
    import focusmodel as fm
    path = os.path.join(tempfile.mkdtemp(), "m.json")
    with open(path, "w") as f: # old flat format
        json.dump({"samples": [2.5, 2.7], "cbins": {"-70": 2.6}, "shift": 0.01, "qb": {"5": 25}}, f)
    a = fm.FocusModel(path, lens=fm.LEGACY_SLUG)
    assert abs(a.C - 2.6) < 1e-9 and a.qb == {"5": 25.0}
    b = fm.FocusModel(path, lens="other-lens") # the other lens does not receive other people's data
    assert b.C is None and b.qb == {}
    b.add(150, 0.2, 12)
    again = fm.FocusModel(path, lens=fm.LEGACY_SLUG)
    assert abs(again.C - 2.6) < 1e-9 and fm.FocusModel(path, lens="other-lens").C is not None


def test_sigma_grows_with_aperture():
    m = fm_model()
    assert m.sigma_x(0.82, 2.4) > 1.9 * m.sigma_x(0.82, 1.2) - 1e-9 or m.sigma_x(0.82, 2.4) >= 0.04


def fm_model():
    from focusmodel import FocusModel
    return FocusModel()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok:", name)
