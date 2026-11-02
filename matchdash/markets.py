"""Market registry.  Each market is a function of a scoreline context -> probability.

To add your own market, append to MARKETS at the bottom of this file, e.g.

    add("home_o15_win", "Home win & Over 1.5", "Combos", lambda c: _p(c, (c.I > c.J) & (c.I + c.J > 1.5)), rank=True)

`c.g` is the full-time scoreline grid (rows = home goals, columns = away goals), `c.I`/`c.J` the
home/away goal indices, `c.lam_h`/`c.lam_a` the expected goals, `c.ht` the half-time structures.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Optional
import numpy as np

_LOGFACT = np.array([math.lgamma(k + 1) for k in range(0, 61)])


def pois_pmf(lam: float, n: int) -> np.ndarray:
    k = np.arange(n + 1)
    if lam <= 1e-12:
        v = np.zeros(n + 1); v[0] = 1.0
        return v
    return np.exp(-lam + k * math.log(lam) - _LOGFACT[: n + 1])


@dataclass
class Ctx:
    g: np.ndarray
    lam_h: float
    lam_a: float
    sh: float          # home share of goals in 1st half
    sa: float          # away share of goals in 1st half
    N: int

    def __post_init__(self):
        self.I = np.arange(self.N + 1)[:, None]
        self.J = np.arange(self.N + 1)[None, :]

    @cached_property
    def ht(self) -> dict:
        N = self.N
        g1 = np.outer(pois_pmf(self.lam_h * self.sh, N), pois_pmf(self.lam_a * self.sa, N))
        g2 = np.outer(pois_pmf(self.lam_h * (1 - self.sh), N), pois_pmf(self.lam_a * (1 - self.sa), N))
        g1, g2 = g1 / g1.sum(), g2 / g2.sum()
        d = np.arange(-N, N + 1)
        p1 = np.array([np.trace(g1, offset=-x) for x in d])
        p2 = np.array([np.trace(g2, offset=-x) for x in d])
        D1, D2 = d[:, None], d[None, :]
        M = p1[:, None] * p2[None, :]
        htft = {(x, y): float(M[(np.sign(D1) == x) & (np.sign(D1 + D2) == y)].sum())
                for x in (1, 0, -1) for y in (1, 0, -1)}
        return {"d": d, "p1": p1, "htft": htft, "lam": self.lam_h * self.sh + self.lam_a * self.sa}


@dataclass
class Market:
    key: str
    label: str
    group: str
    fn: Callable[[Ctx], float]
    rank: bool = True              # include in "strongest / least likely" rankings
    odds: Optional[str] = None     # odds field used by the Value Check (oh/od/oa/oo/ou/by/bn)


MARKETS: list[Market] = []


def add(key, label, group, fn, rank=True, odds=None):
    MARKETS.append(Market(key, label, group, fn, rank, odds))


def _p(c: Ctx, mask) -> float:
    return float(c.g[np.broadcast_to(mask, c.g.shape)].sum())


def _htp(c: Ctx, sign: int) -> float:
    return float(c.ht["p1"][np.sign(c.ht["d"]) == sign].sum())


def _cdf_over(lam: float, line: int, N: int) -> float:
    return float(1.0 - pois_pmf(lam, N)[: line + 1].sum())


# ---- full-time result ----
add("home", "Home win (1)", "Result", lambda c: _p(c, c.I > c.J), True, "oh")
add("draw", "Draw (X)", "Result", lambda c: _p(c, c.I == c.J), True, "od")
add("away", "Away win (2)", "Result", lambda c: _p(c, c.I < c.J), True, "oa")
add("dc1x", "Home or Draw (1X)", "Double chance", lambda c: _p(c, c.I >= c.J), False)
add("dcx2", "Draw or Away (X2)", "Double chance", lambda c: _p(c, c.I <= c.J), False)
add("dc12", "Home or Away (12)", "Double chance", lambda c: _p(c, c.I != c.J), False)
add("dnbh", "Draw No Bet – Home", "DNB", lambda c: _p(c, c.I > c.J) / max(_p(c, c.I != c.J), 1e-12), False)
add("dnba", "Draw No Bet – Away", "DNB", lambda c: _p(c, c.I < c.J) / max(_p(c, c.I != c.J), 1e-12), False)
# ---- half-time ----
add("hth_h", "HT Home lead", "Half-time", lambda c: _htp(c, 1))
add("hth_d", "HT Draw", "Half-time", lambda c: _htp(c, 0))
add("hth_a", "HT Away lead", "Half-time", lambda c: _htp(c, -1))
add("hto05", "HT Over 0.5 goals", "Half-time", lambda c: _cdf_over(c.ht["lam"], 0, c.N))
add("hto15", "HT Over 1.5 goals", "Half-time", lambda c: _cdf_over(c.ht["lam"], 1, c.N))
# ---- goal lines ----
for _t in (0.5, 1.5, 2.5, 3.5, 4.5):
    add(f"o{_t}", f"Over {_t} goals", "Goals", (lambda c, t=_t: _p(c, c.I + c.J > t)),
        _t != 0.5, "oo" if _t == 2.5 else None)
    add(f"u{_t}", f"Under {_t} goals", "Goals", (lambda c, t=_t: _p(c, c.I + c.J < t)),
        _t not in (0.5, 4.5), "ou" if _t == 2.5 else None)
# ---- BTTS ----
add("btts_y", "BTTS – Yes", "BTTS", lambda c: _p(c, (c.I > 0) & (c.J > 0)), True, "by")
add("btts_n", "BTTS – No", "BTTS", lambda c: _p(c, (c.I == 0) | (c.J == 0)), True, "bn")
add("btts_o25", "BTTS Yes & Over 2.5", "BTTS", lambda c: _p(c, (c.I > 0) & (c.J > 0) & (c.I + c.J > 2.5)))
# ---- team goals ----
add("h_o05", "Home team Over 0.5", "Team goals", lambda c: _cdf_over(c.lam_h, 0, c.N), False)
add("h_o15", "Home team Over 1.5", "Team goals", lambda c: _cdf_over(c.lam_h, 1, c.N))
add("a_o05", "Away team Over 0.5", "Team goals", lambda c: _cdf_over(c.lam_a, 0, c.N), False)
add("a_o15", "Away team Over 1.5", "Team goals", lambda c: _cdf_over(c.lam_a, 1, c.N))
add("cs_h", "Home clean sheet", "Team goals", lambda c: _p(c, c.J == 0))
add("cs_a", "Away clean sheet", "Team goals", lambda c: _p(c, c.I == 0))
# ---- HT/FT ----
_NAMES = {1: "Home", 0: "Draw", -1: "Away"}
for _x in (1, 0, -1):
    for _y in (1, 0, -1):
        add(f"hf_{_NAMES[_x]}_{_NAMES[_y]}", f"HT {_NAMES[_x]} / FT {_NAMES[_y]}", "HT/FT",
            (lambda c, x=_x, y=_y: c.ht["htft"][(x, y)]), False)

# ======================= YOUR OWN MARKETS GO HERE =======================
# add("home_o15", "Home win & Over 1.5", "Combos", lambda c: _p(c, (c.I > c.J) & (c.I + c.J > 1.5)))
# (new groups must also be listed in config.PANELS to appear on the dashboard)
# ========================================================================

MARKET_BY_KEY = {m.key: m for m in MARKETS}
