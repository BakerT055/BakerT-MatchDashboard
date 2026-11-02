"""Walk-forward backtest: fit on matches before each period, predict that period, score the result."""
from __future__ import annotations
from dataclasses import replace
import numpy as np
import pandas as pd
from .config import Settings
from .model import Snapshot, predict

KEYS = {"home", "draw", "away", "o2.5", "btts_y"}
EPS = 1e-12


def walk_forward(history: pd.DataFrame, s: Settings, start, end=None, step_days: int = 30) -> pd.DataFrame:
    d = history[history["FTHG"].notna() & history["FTAG"].notna() & history["Date"].notna()]
    test = d[d["Date"] >= pd.Timestamp(start)]
    if end is not None:
        test = test[test["Date"] <= pd.Timestamp(end)]
    if test.empty:
        return pd.DataFrame()
    t0 = test["Date"].min().normalize()
    period = ((test["Date"] - t0).dt.days // step_days)
    rows = []
    for p, g in test.groupby(period):
        asof = t0 + pd.Timedelta(days=int(p) * step_days)
        snap = Snapshot(d, s, asof=asof)
        if snap.scopes["ALL"].n == 0:
            continue
        for r in g.itertuples():
            pr = predict(snap, r.HomeTeam, r.AwayTeam, r.League, keys=KEYS)
            m = pr.markets
            rows.append({"Date": r.Date, "League": r.League, "Home": r.HomeTeam, "Away": r.AwayTeam,
                         "pH": m["home"], "pD": m["draw"], "pA": m["away"], "pO25": m["o2.5"], "pBTTS": m["btts_y"],
                         "res": 0 if r.FTHG > r.FTAG else (1 if r.FTHG == r.FTAG else 2),
                         "over": int(r.FTHG + r.FTAG > 2.5), "btts": int(r.FTHG > 0 and r.FTAG > 0)})
    return pd.DataFrame(rows)


def _bin_metrics(p: np.ndarray, y: np.ndarray) -> dict:
    p = np.clip(p, EPS, 1 - EPS)
    return {"Log loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "Brier": float(np.mean((p - y) ** 2)),
            "Accuracy": float(np.mean((p > 0.5) == (y == 1)))}


def metrics(bt: pd.DataFrame) -> pd.DataFrame:
    if bt.empty:
        return pd.DataFrame()
    P = bt[["pH", "pD", "pA"]].to_numpy(); P = np.clip(P / P.sum(axis=1, keepdims=True), EPS, 1)
    y = bt["res"].to_numpy(); Y = np.eye(3)[y]
    res = {"Log loss": float(-np.mean(np.log(P[np.arange(len(y)), y]))), "Brier": float(np.mean(np.sum((P - Y) ** 2, axis=1))),
           "Accuracy": float(np.mean(P.argmax(axis=1) == y))}
    freq = np.bincount(y, minlength=3) / len(y)
    ref = {"Log loss": float(-np.mean(np.log(np.clip(freq[y], EPS, 1)))), "Brier": float(np.mean(np.sum((freq - Y) ** 2, axis=1))),
           "Accuracy": float(freq.max())}
    over = _bin_metrics(bt["pO25"].to_numpy(), bt["over"].to_numpy()); bt_ = _bin_metrics(bt["pBTTS"].to_numpy(), bt["btts"].to_numpy())
    def refb(y_):
        f = y_.mean(); return _bin_metrics(np.full(len(y_), f), y_.to_numpy())
    out = pd.DataFrame({"Result 1X2": res, "Result (reference)": ref, "Over 2.5": over, "Over 2.5 (reference)": refb(bt["over"]),
                        "BTTS": bt_, "BTTS (reference)": refb(bt["btts"])}).T
    out.insert(0, "Matches", len(bt))
    return out


def calibration(bt: pd.DataFrame, col: str, outcome: str, bins: int = 10) -> pd.DataFrame:
    if bt.empty:
        return pd.DataFrame()
    edges = np.linspace(0, 1, bins + 1)
    b = pd.cut(bt[col], edges, include_lowest=True)
    g = bt.groupby(b, observed=True).agg(Matches=(col, "size"), Predicted=(col, "mean"), Actual=(outcome, "mean"))
    g.index = g.index.astype(str)
    return g


def sweep(history, s: Settings, param: str, values, start, end=None, step_days=30) -> pd.DataFrame:
    rows = []
    for v in values:
        bt = walk_forward(history, replace(s, **{param: v}), start, end, step_days)
        if bt.empty:
            continue
        m = metrics(bt)
        rows.append({param: v, "Matches": len(bt), "1X2 log loss": m.loc["Result 1X2", "Log loss"],
                     "1X2 accuracy": m.loc["Result 1X2", "Accuracy"], "O2.5 log loss": m.loc["Over 2.5", "Log loss"],
                     "BTTS log loss": m.loc["BTTS", "Log loss"]})
    return pd.DataFrame(rows)
