"""Run:  python tests/test_ml.py   – tests for the ML layer (needs scikit-learn)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from matchdash import data, model, ml, engine, backtest
from matchdash.config import Settings

ROOT = Path(__file__).resolve().parent.parent
hist = data.load_history([ROOT / "data" / "sample_history.csv"])
PREFS = ["Generic"]


def test_no_leakage():
    """Features for early matches must be identical whether or not later matches exist."""
    X_full, m_full = ml.FeatureBuilder(PREFS).fit_transform(hist)
    cut = 600
    X_part, m_part = ml.FeatureBuilder(PREFS).fit_transform(hist.iloc[:cut])
    a = X_full.loc[X_part.index].to_numpy(); b = X_part.to_numpy()
    assert np.allclose(a, b, equal_nan=True), "future data leaked into earlier features"
    # and a match's own result is never in its own features
    h2 = hist.copy(); i = 700
    h2.loc[i, ["FTHG", "FTAG"]] = [9, 9]
    X2, _ = ml.FeatureBuilder(PREFS).fit_transform(h2)
    ix = X_full.index[X_full.index.get_loc(i)] if i in X_full.index else None
    assert ix is not None and np.allclose(X_full.loc[ix].to_numpy(), X2.loc[ix].to_numpy(), equal_nan=True)


def test_fit_grid_hits_targets():
    s = Settings(); g = model.make_grid(1.6, 1.1, s)
    out = model.fit_grid(g, 0.50, 0.25, 0.25, 0.62, 0.55)
    p, _ = model.markets_from_grid(out, 1.6, 1.1, 0.45, 0.45)
    assert abs(out.sum() - 1) < 1e-9
    for k, t in [("home", 0.50), ("draw", 0.25), ("away", 0.25), ("o2.5", 0.62), ("btts_y", 0.55)]:
        assert abs(p[k] - t) < 2e-3, (k, p[k], t)


def test_train_predict_engine():
    D = pd.Timestamp("2026-11-07")
    s = Settings(); snap = model.Snapshot(hist, s)
    b = ml.train_bundle(hist, PREFS, use_odds=False, calibrate=True)
    o = ml.ml_predict(b, "Team A", "Team B", "Sample League", pd.Timestamp("2026-11-07"))
    assert abs(o["pH"] + o["pD"] + o["pA"] - 1) < 1e-9 and 0 < o["lam_h"] < 6
    base = engine.predict_any(snap, b, Settings(engine="Poisson"), "Team A", "Team B", "Sample League", D)
    mlp = engine.predict_any(snap, b, Settings(engine="ML"), "Team A", "Team B", "Sample League", D)
    bl = engine.predict_any(snap, b, Settings(engine="Blend", ml_blend=0.5), "Team A", "Team B", "Sample League")
    assert base.source == "Poisson" and mlp.source == "ML" and bl.source.startswith("Blend")
    for pr in (mlp, bl):
        m = pr.markets
        assert abs(m["home"] + m["draw"] + m["away"] - 1) < 1e-9
        assert abs(sum(v for k, v in m.items() if k.startswith("hf_")) - 1) < 1e-9
        assert abs(pr.grid.sum() - 1) < 1e-9
    assert abs(mlp.markets["home"] - o["pH"]) < 5e-3 and abs(mlp.markets["o2.5"] - o["pO"]) < 5e-3
    lo, hi = sorted([base.markets["home"], mlp.markets["home"]])
    assert lo - 5e-3 <= bl.markets["home"] <= hi + 5e-3
    nob = ml.train_bundle(hist, PREFS, use_odds=True, calibrate=False)
    w = engine.predict_any(snap, nob, Settings(engine="ML"), "Team A", "Team B", "Sample League")
    assert any("odds" in x for x in w.warnings)


def test_walk_forward_and_compare():
    s = Settings(); t = time.time()
    wf = ml.ml_walk_forward(hist, "2025-08-01", step_days=120, calibrate=True, prefs=PREFS)
    pw = backtest.walk_forward(hist, s, "2025-08-01", None, 120)
    all_t, bk_t, d = ml.compare_metrics(wf, pw)
    assert len(d) > 300 and "ML" in all_t.index and "Poisson (statistical)" in all_t.index
    assert not bk_t.empty and "Bookmaker (margin removed)" in bk_t.index
    print(f"  walk-forward {len(d)} matches in {time.time()-t:.1f}s")
    print(all_t.round(4).to_string()); print(bk_t.round(4).to_string())
    imp = ml.feature_importance(hist, PREFS)
    print("  top features:", ", ".join(imp.head(6)["Feature"]))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
