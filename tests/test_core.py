"""Run:  python tests/test_core.py   (or pytest).  Checks the engine against the Excel workbook's results."""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from matchdash import data, model, h2h, backtest
from matchdash.config import Settings
from matchdash.markets import MARKET_BY_KEY

ROOT = Path(__file__).resolve().parent.parent
hist = data.load_history([ROOT / "data" / "sample_history.csv"])
fx = data.load_fixtures([ROOT / "data" / "sample_fixtures.csv"])


def test_load():
    assert len(hist) == 1140 and hist["League"].nunique() == 1
    assert {"oh__Generic", "oo__Generic"} <= set(hist.columns)
    assert len(fx) == 8 and fx["oh__Generic"].notna().all()


def test_matches_excel():
    s = Settings()
    snap = model.Snapshot(hist, s)
    p = model.predict(snap, "Team A", "Team B", "Sample League")
    # values produced by the Excel workbook for the same data / settings
    assert abs(p.lam_h - 2.3451) < 5e-4 and abs(p.lam_a - 0.7426) < 5e-4, (p.lam_h, p.lam_a)
    assert abs(p.markets["home"] - 0.73228) < 1e-4
    assert abs(p.markets["o2.5"] - 0.59614) < 1e-4
    assert abs(p.markets["home"] + p.markets["draw"] + p.markets["away"] - 1) < 1e-9
    assert abs(p.markets["o2.5"] + p.markets["u2.5"] - 1) < 1e-9
    assert abs(sum(v for k, v in p.markets.items() if k.startswith("hf_")) - 1) < 1e-9
    assert abs(p.markets["hth_h"] + p.markets["hth_d"] + p.markets["hth_a"] - 1) < 1e-9


def test_h2h_matches_excel():
    s = Settings(); snap = model.Snapshot(hist, s)
    hc = h2h.history_check(snap, hist, "Team A", "Team B", "Sample League")
    assert hc.loc["Games", "Home @ home"] == 57 and hc.loc["Games", "H2H (all history)"] == 6
    assert abs(hc.loc["Win %", "H2H (all history)"] - 0.6667) < 1e-3
    assert abs(hc.loc["Over 2.5 %", "H2H (all history)"] - 0.8333) < 1e-3
    assert abs(hc.loc["Win %", "Home @ home"] - 0.5439) < 1e-3
    rt = h2h.recent_table(hist, "Team A", "Team B", 8, ["Generic"])
    assert len(rt) == 6 and rt.iloc[0]["Date"] == "07-Jan-2026" and rt.iloc[0]["Home"] == "Team B"


def test_options():
    base = model.predict(model.Snapshot(hist, Settings()), "Team A", "Team B", "Sample League")
    dc = model.predict(model.Snapshot(hist, Settings(dc_rho=-0.1)), "Team A", "Team B", "Sample League")
    hl = model.predict(model.Snapshot(hist, Settings(half_life_days=365)), "Team A", "Team B", "Sample League")
    assert dc.markets["draw"] != base.markets["draw"] and abs(sum(dc.grid.flat) - 1) < 1e-9
    assert hl.lam_h != base.lam_h
    unk = model.predict(model.Snapshot(hist, Settings()), "Nobody", "Team B", "Sample League")
    assert unk.warnings and unk.lam_h > 0


def test_backtest():
    s = Settings()
    bt = backtest.walk_forward(hist, s, "2025-08-01", step_days=60)
    m = backtest.metrics(bt)
    assert len(bt) > 300 and m.loc["Result 1X2", "Log loss"] < m.loc["Result (reference)", "Log loss"]
    sw = backtest.sweep(hist, s, "shrinkage", [2, 5, 20], "2025-08-01", step_days=90)
    assert len(sw) == 3


def test_names_check():
    bad = fx.copy(); bad.loc[0, "HomeTeam"] = "Team AA"
    chk = data.check_team_names(bad, hist)
    assert len(chk) == 1 and "Team A" in chk.iloc[0]["Suggestions"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
