"""Poisson match model – same logic as the Excel dashboard, plus optional recency weighting
and Dixon-Coles low-score correction (both off by default so results match the workbook)."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd

from .config import Settings
from .markets import MARKETS, Ctx, pois_pmf


class Scope:
    """Aggregated team / league statistics for one league (or all data pooled)."""

    def __init__(self, g: pd.DataFrame):
        w = g["w"]
        self.n = len(g)
        self.sw = float(w.sum())
        self.hg = float((w * g["FTHG"]).sum())
        self.ag = float((w * g["FTAG"]).sum())
        gg = g.assign(wh=w * g["FTHG"], wa=w * g["FTAG"])
        h = gg.groupby("HomeTeam").agg(n=("w", "sum"), games=("w", "size"), gf=("wh", "sum"), ga=("wa", "sum"))
        a = gg.groupby("AwayTeam").agg(n=("w", "sum"), games=("w", "size"), gf=("wa", "sum"), ga=("wh", "sum"))
        self.home = h.to_dict("index")
        self.away = a.to_dict("index")
        ht = g[g["HTHG"].notna() & g["HTAG"].notna()]
        self.ht_h = float((ht["w"] * ht["HTHG"]).sum()); self.ft_h = float((ht["w"] * ht["FTHG"]).sum())
        self.ht_a = float((ht["w"] * ht["HTAG"]).sum()); self.ft_a = float((ht["w"] * ht["FTAG"]).sum())


class Snapshot:
    """Model fitted on history up to `asof` (exclusive).  Build once, predict many times."""

    def __init__(self, history: pd.DataFrame, s: Settings, asof: Optional[pd.Timestamp] = None):
        d = history[history["FTHG"].notna() & history["FTAG"].notna()]
        if s.history_from:
            d = d[d["Date"] >= pd.Timestamp(s.history_from)]
        if asof is not None:
            d = d[d["Date"] < asof]
        ref = asof if asof is not None else (d["Date"].max() if len(d) and d["Date"].notna().any() else pd.Timestamp.today())
        if s.half_life_days and s.half_life_days > 0:
            age = (ref - d["Date"]).dt.days.clip(lower=0)
            w = np.power(0.5, age / s.half_life_days).fillna(1.0)
        else:
            w = pd.Series(1.0, index=d.index)
        d = d.assign(w=w)
        self.s = s
        self.data = d
        self.scopes = {"ALL": Scope(d)}
        for lg, g in d.groupby("League"):
            self.scopes[lg] = Scope(g)

    def scope_key(self, league: Optional[str]) -> str:
        return league if (self.s.only_fixture_league and league in self.scopes) else "ALL"

    def league_data(self, league: Optional[str]) -> pd.DataFrame:
        return self.data[self.data["League"] == league] if self.scope_key(league) != "ALL" else self.data


@dataclass
class Prediction:
    home: str
    away: str
    league: Optional[str]
    lam_h: float
    lam_a: float
    ht_share: tuple
    grid: np.ndarray
    markets: dict
    top_scores: list
    n_matches: float
    n_home: float
    n_away: float
    games_home: int
    games_away: int
    warnings: list = field(default_factory=list)
    source: str = "Poisson"

    @property
    def total_xg(self) -> float:
        return self.lam_h + self.lam_a


def make_grid(lam_h: float, lam_a: float, s: Settings) -> np.ndarray:
    N = int(s.max_goals)
    g = np.outer(pois_pmf(lam_h, N), pois_pmf(lam_a, N))
    if s.dc_rho:
        r = s.dc_rho
        g[0, 0] *= max(0.0, 1 - lam_h * lam_a * r); g[0, 1] *= max(0.0, 1 + lam_h * r)
        g[1, 0] *= max(0.0, 1 + lam_a * r);         g[1, 1] *= max(0.0, 1 - r)
    return g / g.sum()


def markets_from_grid(g: np.ndarray, lam_h: float, lam_a: float, sh: float, sa: float, keys: Optional[set] = None):
    """All registered markets + top-5 scores from a scoreline grid."""
    N = g.shape[0] - 1
    ctx = Ctx(g=g, lam_h=lam_h, lam_a=lam_a, sh=sh, sa=sa, N=N)
    probs = {m.key: m.fn(ctx) for m in MARKETS if keys is None or m.key in keys}
    flat = np.argsort(g, axis=None)[::-1][:5]
    top = [(int(i // (N + 1)), int(i % (N + 1)), float(g.flat[i])) for i in flat]
    return probs, top


def fit_grid(g: np.ndarray, p_home: float, p_draw: float, p_away: float, p_over25: float, p_btts: float,
             iters: int = 60) -> np.ndarray:
    """Iterative proportional fitting: rescale scoreline cells so the grid reproduces the target
    1X2, Over/Under 2.5 and BTTS probabilities while keeping the score structure coherent."""
    N = g.shape[0] - 1
    I = np.arange(N + 1)[:, None]; J = np.arange(N + 1)[None, :]
    bt = np.broadcast_to((I > 0) & (J > 0), g.shape)
    parts = [
        ([I > J, I == J, I < J], [p_home, p_draw, p_away]),
        ([I + J > 2.5, I + J < 2.5], [p_over25, 1 - p_over25]),
        ([bt, ~bt], [p_btts, 1 - p_btts]),
    ]
    g = g.copy()
    for _ in range(iters):
        for masks, targets in parts:
            for m, t in zip(masks, targets):
                m = np.broadcast_to(m, g.shape)
                cur = g[m].sum()
                if cur > 0:
                    g[m] *= max(t, 0.0) / cur
        g /= g.sum()
    return g


def predict(snap: Snapshot, home: str, away: str, league: Optional[str] = None, keys: Optional[set] = None) -> Prediction:
    s = snap.s
    sc = snap.scopes[snap.scope_key(league)]
    warns = []
    if sc.sw <= 0:
        lgh, lga = 1.45, 1.15
        warns.append("No valid matches in History for this selection – using generic league averages.")
    else:
        lgh, lga = sc.hg / sc.sw, sc.ag / sc.sw
    if s.only_fixture_league and league not in snap.scopes:
        warns.append(f"League '{league}' not found in History – all leagues pooled.")

    hs = sc.home.get(home); as_ = sc.away.get(away)
    nH, gfH, gaH, gmH = (hs["n"], hs["gf"], hs["ga"], int(hs["games"])) if hs else (0.0, 0.0, 0.0, 0)
    nA, gfA, gaA, gmA = (as_["n"], as_["gf"], as_["ga"], int(as_["games"])) if as_ else (0.0, 0.0, 0.0, 0)
    k = s.shrinkage

    def rel(gf, n, lg):
        return ((gf + k * lg) / (n + k)) / lg if lg > 0 and (n + k) > 0 else 1.0

    hatt, hdef = rel(gfH, nH, lgh), rel(gaH, nH, lga)
    aatt, adef = rel(gfA, nA, lga), rel(gaA, nA, lgh)
    lam_h = hatt * adef * lgh
    lam_a = aatt * hdef * lga

    sh = sc.ht_h / sc.ft_h if sc.ft_h > 0 else s.ht_share_fallback
    sa = sc.ht_a / sc.ft_a if sc.ft_a > 0 else s.ht_share_fallback
    sh, sa = min(1.0, max(0.0, sh)), min(1.0, max(0.0, sa))

    g = make_grid(lam_h, lam_a, s)
    probs, top = markets_from_grid(g, lam_h, lam_a, sh, sa, keys)

    if sc.n < s.min_matches_warn:
        warns.append(f"Only {sc.n} valid matches in the model – add more history for reliable output.")
    if min(gmH, gmA) < s.min_games_warn:
        warns.append(f"Low sample: {home} {gmH} home games · {away} {gmA} away games – treat with caution.")
    return Prediction(home, away, league, lam_h, lam_a, (sh, sa), g, probs, top, sc.n, nH, nA, gmH, gmA, warns)
