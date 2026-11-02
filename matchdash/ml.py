"""Machine-learning layer.

* FeatureBuilder – walks through history in date order and builds, for every match, features that use
  ONLY information available before kick-off (Elo, recent form, venue-specific goal rates, rest days,
  head-to-head, league averages, optionally bookmaker-implied probabilities).  This prevents leakage.
* Models – gradient-boosted trees: Poisson-loss goal-rate regressors (home / away) and (optionally
  calibrated) classifiers for the result (1X2), Over 2.5 and BTTS.
* ml_walk_forward / compare_metrics – out-of-sample comparison against the Poisson model and bookmaker odds.

Be sceptical of any football model, including this one: bookmaker odds are the benchmark to beat.
"""
from __future__ import annotations
import math
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import DATA_DIR
from .data import resolve_odds

NAN = float("nan")
BUNDLE_FILE = DATA_DIR / "ml_bundle.pkl"
KEEP = 12  # matches of history kept per team list


def ml_available() -> bool:
    try:
        import sklearn  # noqa: F401
        return True
    except ImportError:
        return False


# ============================================================ feature engineering
def _lm(lst, k):
    return float(np.mean(lst[-k:])) if lst else NAN


class _Team:
    def __init__(self):
        self.gf, self.ga, self.pts = [], [], []
        self.hgf, self.hga, self.hpts = [], [], []
        self.agf, self.aga, self.apts = [], [], []
        self.last = None
        self.n = self.nh = self.na = 0
        self.ewm = [NAN, NAN]; self.ewm_h = [NAN, NAN]; self.ewm_a = [NAN, NAN]


def _push(lst, x):
    lst.append(x)
    if len(lst) > KEEP:
        lst.pop(0)


def _ewm_update(e, n, gf, ga):
    a = max(1.0 / n, 0.07)
    e[0] = gf if e[0] != e[0] else e[0] + a * (gf - e[0])
    e[1] = ga if e[1] != e[1] else e[1] + a * (ga - e[1])


def _pts(gf, ga):
    return 3.0 if gf > ga else (1.0 if gf == ga else 0.0)


SIDE_FEATS = ["n", "rest", "gf5", "ga5", "ppg5", "gf10", "ga10", "ppg10", "ewm_gf", "ewm_ga",
              "v_gf5", "v_ga5", "v_ppg5", "v_ewm_gf", "v_ewm_ga"]
BASE_FEATURES = (["elo_h", "elo_a", "elo_diff", "elo_exp"] + [f"{f}_h" for f in SIDE_FEATS] + [f"{f}_a" for f in SIDE_FEATS] +
                 ["lg_h_avg", "lg_a_avg", "lg_over", "lg_btts", "xg_h_simple", "xg_a_simple",
                  "h2h_n", "h2h_ppg", "h2h_goals"])
ODDS_FEATURES = ["imp_h", "imp_d", "imp_a", "imp_over"]


def implied(odds: Optional[dict]):
    """Margin-removed bookmaker probabilities (proportional method)."""
    out = {"h": NAN, "d": NAN, "a": NAN, "o": NAN}
    if not odds:
        return out
    o = [odds.get(k) for k in ("oh", "od", "oa")]
    if all(v is not None and v == v and v > 1 for v in o):
        inv = [1 / v for v in o]; t = sum(inv)
        out["h"], out["d"], out["a"] = (x / t for x in inv)
    oo, ou = odds.get("oo"), odds.get("ou")
    if oo is not None and ou is not None and oo == oo and ou == ou and oo > 1 and ou > 1:
        out["o"] = (1 / oo) / (1 / oo + 1 / ou)
    return out


class FeatureBuilder:
    def __init__(self, prefs=("B365", "PS", "Avg", "Max", "Generic"), use_odds: bool = False,
                 elo_k: float = 20.0, elo_home_adv: float = 60.0):
        self.prefs = list(prefs); self.use_odds = use_odds
        self.elo_k, self.elo_ha = elo_k, elo_home_adv
        self.teams: dict = {}; self.elo: dict = {}; self.lg: dict = {}; self.h2h: dict = {}
        self.last_date = None; self.n_matches = 0

    @property
    def features(self) -> list:
        return BASE_FEATURES + (ODDS_FEATURES if self.use_odds else [])

    # ---- per-side features
    def _side(self, t, venue, date):
        st = self.teams.get(t)
        d = {f: NAN for f in SIDE_FEATS}
        d["n"] = 0
        if st is None:
            return d
        vgf, vga, vpts, vew = (st.hgf, st.hga, st.hpts, st.ewm_h) if venue == "h" else (st.agf, st.aga, st.apts, st.ewm_a)
        d["n"] = st.n
        if st.last is not None and date is not None and date == date:
            d["rest"] = float(min((date - st.last).days, 30))
        d.update(gf5=_lm(st.gf, 5), ga5=_lm(st.ga, 5), ppg5=_lm(st.pts, 5), gf10=_lm(st.gf, 10), ga10=_lm(st.ga, 10),
                 ppg10=_lm(st.pts, 10), ewm_gf=st.ewm[0], ewm_ga=st.ewm[1],
                 v_gf5=_lm(vgf, 5), v_ga5=_lm(vga, 5), v_ppg5=_lm(vpts, 5), v_ewm_gf=vew[0], v_ewm_ga=vew[1])
        return d

    def row_features(self, h, a, league, date, odds=None) -> dict:
        eh, ea = self.elo.get(h, 1500.0), self.elo.get(a, 1500.0)
        exp = 1 / (1 + 10 ** (-((eh + self.elo_ha) - ea) / 400))
        f = {"elo_h": eh, "elo_a": ea, "elo_diff": eh - ea, "elo_exp": exp}
        sh, sa = self._side(h, "h", date), self._side(a, "a", date)
        f.update({f"{k}_h": v for k, v in sh.items()}); f.update({f"{k}_a": v for k, v in sa.items()})
        lg = self.lg.get(league)
        if lg and lg[0] >= 30:
            n = lg[0]; lh, la, lo, lb = lg[1] / n, lg[2] / n, lg[3] / n, lg[4] / n
        else:
            lh, la, lo, lb = 1.45, 1.15, 0.5, 0.5
        f.update(lg_h_avg=lh, lg_a_avg=la, lg_over=lo, lg_btts=lb)
        f["xg_h_simple"] = sh["v_ewm_gf"] * sa["v_ewm_ga"] / lh if lh > 0 else NAN
        f["xg_a_simple"] = sa["v_ewm_gf"] * sh["v_ewm_ga"] / la if la > 0 else NAN
        m = self.h2h.get(frozenset((h, a)), [])
        f["h2h_n"] = len(m)
        if m:
            pts, tg = [], []
            for hm, hg, ag in m[-6:]:
                gf, ga = (hg, ag) if hm == h else (ag, hg)
                pts.append(_pts(gf, ga)); tg.append(hg + ag)
            f["h2h_ppg"], f["h2h_goals"] = float(np.mean(pts)), float(np.mean(tg))
        else:
            f["h2h_ppg"] = f["h2h_goals"] = NAN
        if self.use_odds:
            im = implied(odds)
            f.update(imp_h=im["h"], imp_d=im["d"], imp_a=im["a"], imp_over=im["o"])
        return f

    # ---- state update after a result is known
    def update(self, h, a, league, date, hg, ag):
        eh, ea = self.elo.get(h, 1500.0), self.elo.get(a, 1500.0)
        exp = 1 / (1 + 10 ** (-((eh + self.elo_ha) - ea) / 400))
        s_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        gd = abs(hg - ag)
        mult = 1.0 if gd <= 1 else (1.5 if gd == 2 else (11 + gd) / 8)
        delta = self.elo_k * mult * (s_h - exp)
        self.elo[h], self.elo[a] = eh + delta, ea - delta
        for t, gf, ga, venue in ((h, hg, ag, "h"), (a, ag, hg, "a")):
            st = self.teams.setdefault(t, _Team())
            st.n += 1; st.last = date
            _push(st.gf, gf); _push(st.ga, ga); _push(st.pts, _pts(gf, ga))
            _ewm_update(st.ewm, st.n, gf, ga)
            if venue == "h":
                st.nh += 1; _push(st.hgf, gf); _push(st.hga, ga); _push(st.hpts, _pts(gf, ga)); _ewm_update(st.ewm_h, st.nh, gf, ga)
            else:
                st.na += 1; _push(st.agf, gf); _push(st.aga, ga); _push(st.apts, _pts(gf, ga)); _ewm_update(st.ewm_a, st.na, gf, ga)
        lg = self.lg.setdefault(league, [0, 0.0, 0.0, 0.0, 0.0])
        lg[0] += 1; lg[1] += hg; lg[2] += ag; lg[3] += float(hg + ag > 2.5); lg[4] += float(hg > 0 and ag > 0)
        self.h2h.setdefault(frozenset((h, a)), []).append((h, hg, ag))
        if len(self.h2h[frozenset((h, a))]) > KEEP:
            self.h2h[frozenset((h, a))].pop(0)
        self.last_date = date if (self.last_date is None or date > self.last_date) else self.last_date
        self.n_matches += 1

    def fit_transform(self, history: pd.DataFrame):
        """Pre-match features for every historical match (chronological, leakage-free)."""
        df = resolve_odds(history, self.prefs)
        df = df[df["Date"].notna() & df["FTHG"].notna() & df["FTAG"].notna()].sort_values("Date", kind="stable")
        cols = ["Date", "League", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "oh", "od", "oa", "oo", "ou"]
        feats, bk = [], []
        for r in df[cols].itertuples(index=False):
            date, lg, h, a, hg, ag, oh, od, oa, oo, ou = r
            odds = {"oh": oh, "od": od, "oa": oa, "oo": oo, "ou": ou}
            feats.append(self.row_features(h, a, lg, date, odds))
            im = implied(odds); bk.append((im["h"], im["d"], im["a"], im["o"]))
            self.update(h, a, lg, date, hg, ag)
        X = pd.DataFrame(feats, columns=self.features, index=df.index).astype(float)
        meta = df[["Date", "League", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].copy()
        meta["res"] = np.where(meta["FTHG"] > meta["FTAG"], 0, np.where(meta["FTHG"] == meta["FTAG"], 1, 2))
        meta["over"] = ((meta["FTHG"] + meta["FTAG"]) > 2.5).astype(int)
        meta["btts"] = ((meta["FTHG"] > 0) & (meta["FTAG"] > 0)).astype(int)
        bk = np.array(bk, dtype=float).reshape(-1, 4)
        for i, c in enumerate(["bH", "bD", "bA", "bO"]):
            meta[c] = bk[:, i]
        return X, meta

    def transform_fixture(self, home, away, league, date=None, odds=None) -> pd.DataFrame:
        date = pd.Timestamp(date) if date is not None and date == date else None
        return pd.DataFrame([self.row_features(home, away, league, date, odds)], columns=self.features).astype(float)


# ============================================================ models
def _models(calibrate: bool, seed: int = 0):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    def reg():
        return HistGradientBoostingRegressor(loss="poisson", max_iter=200, learning_rate=0.04, max_depth=4,
                                             min_samples_leaf=40, l2_regularization=2.0, random_state=seed)

    def clf():
        base = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=3, min_samples_leaf=40,
                                              l2_regularization=2.0, random_state=seed)
        return CalibratedClassifierCV(base, method="sigmoid", cv=3) if calibrate else base
    return {"lam_h": reg(), "lam_a": reg(), "res": clf(), "over": clf(), "btts": clf()}


def fit_models(X: pd.DataFrame, meta: pd.DataFrame, calibrate: bool = True) -> dict:
    m = _models(calibrate)
    m["lam_h"].fit(X, meta["FTHG"]); m["lam_a"].fit(X, meta["FTAG"])
    m["res"].fit(X, meta["res"]); m["over"].fit(X, meta["over"]); m["btts"].fit(X, meta["btts"])
    return m


def _p_class(model, X, cls: int, n_classes: int = 2):
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    return proba[:, classes.index(cls)] if cls in classes else np.zeros(len(X))


def predict_models(models: dict, X: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=X.index)
    out["lam_h"] = np.clip(models["lam_h"].predict(X), 0.05, 6.0)
    out["lam_a"] = np.clip(models["lam_a"].predict(X), 0.05, 6.0)
    for c, k in zip(("pH", "pD", "pA"), (0, 1, 2)):
        out[c] = _p_class(models["res"], X, k)
    s = out[["pH", "pD", "pA"]].sum(axis=1).replace(0, 1)
    for c in ("pH", "pD", "pA"):
        out[c] = np.clip(out[c] / s, 0.005, 0.99)
    s = out[["pH", "pD", "pA"]].sum(axis=1)
    for c in ("pH", "pD", "pA"):
        out[c] = out[c] / s
    out["pO"] = np.clip(_p_class(models["over"], X, 1), 0.01, 0.99)
    out["pB"] = np.clip(_p_class(models["btts"], X, 1), 0.01, 0.99)
    return out


# ============================================================ bundle (trained model + feature state)
@dataclass
class MLBundle:
    fb: FeatureBuilder
    models: dict
    meta: dict = field(default_factory=dict)

    def save(self, path: Path = BUNDLE_FILE):
        DATA_DIR.mkdir(exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path = BUNDLE_FILE) -> Optional["MLBundle"]:
        try:
            if Path(path).exists():
                with open(path, "rb") as f:
                    return pickle.load(f)
        except Exception:
            return None
        return None


def train_bundle(history: pd.DataFrame, prefs, use_odds: bool = False, calibrate: bool = True, min_rows: int = 500) -> MLBundle:
    if not ml_available():
        raise ImportError("scikit-learn is required for the ML engine: pip install scikit-learn")
    fb = FeatureBuilder(prefs=prefs, use_odds=use_odds)
    X, meta = fb.fit_transform(history)
    if len(X) < min_rows:
        raise ValueError(f"Need at least {min_rows} dated matches to train (found {len(X)}). More seasons will help a lot.")
    models = fit_models(X, meta, calibrate)
    info = {"trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "n_rows": len(X), "use_odds": use_odds,
            "calibrate": calibrate, "last_date": fb.last_date, "features": fb.features}
    return MLBundle(fb, models, info)


def ml_predict(bundle: MLBundle, home, away, league, date=None, odds=None) -> dict:
    X = bundle.fb.transform_fixture(home, away, league, date, odds)
    return predict_models(bundle.models, X).iloc[0].to_dict()


# ============================================================ evaluation
def ml_walk_forward(history: pd.DataFrame, start, step_days: int = 120, calibrate: bool = True, use_odds: bool = False,
                    prefs=("B365", "PS", "Avg", "Max", "Generic"), min_train: int = 500) -> pd.DataFrame:
    """Train only on matches before each period, predict that period."""
    fb = FeatureBuilder(prefs=prefs, use_odds=use_odds)
    X, meta = fb.fit_transform(history)
    test_mask = meta["Date"] >= pd.Timestamp(start)
    if not test_mask.any():
        return pd.DataFrame()
    t0 = meta.loc[test_mask, "Date"].min().normalize()
    period = (meta["Date"] - t0).dt.days // step_days
    outs = []
    for p in sorted(set(period[test_mask])):
        asof = t0 + pd.Timedelta(days=int(p) * step_days)
        tr = (meta["Date"] < asof).to_numpy()
        te = (test_mask & (period == p)).to_numpy()
        if tr.sum() < min_train or te.sum() == 0:
            continue
        models = fit_models(X[tr], meta[tr], calibrate)
        pr = predict_models(models, X[te])
        o = meta[te][["Date", "League", "HomeTeam", "AwayTeam", "res", "over", "btts", "bH", "bD", "bA", "bO"]].copy()
        o = o.join(pr.rename(columns={"pH": "mH", "pD": "mD", "pA": "mA", "pO": "mO", "pB": "mB", "lam_h": "mlh", "lam_a": "mla"}))
        outs.append(o)
    return pd.concat(outs).rename(columns={"HomeTeam": "Home", "AwayTeam": "Away"}) if outs else pd.DataFrame()


def _score(pH, pD, pA, pO, pB, res, over, btts) -> dict:
    eps = 1e-12
    P = np.clip(np.column_stack([pH, pD, pA]), eps, 1); P = P / P.sum(axis=1, keepdims=True)
    n = len(res); Y = np.eye(3)[res]
    po, pb = np.clip(pO, eps, 1 - eps), np.clip(pB, eps, 1 - eps)
    return {"1X2 log loss": float(-np.mean(np.log(P[np.arange(n), res]))),
            "1X2 Brier": float(np.mean(np.sum((P - Y) ** 2, axis=1))),
            "1X2 accuracy": float(np.mean(P.argmax(axis=1) == res)),
            "O2.5 log loss": float(-np.mean(over * np.log(po) + (1 - over) * np.log(1 - po))),
            "BTTS log loss": float(-np.mean(btts * np.log(pb) + (1 - btts) * np.log(1 - pb)))}


def compare_metrics(ml_wf: pd.DataFrame, pois_wf: pd.DataFrame, weights=(0.25, 0.5, 0.75)):
    """Returns (all_matches_table, bookmaker_subset_table, merged_frame)."""
    key = ["Date", "League", "Home", "Away"]
    d = ml_wf.merge(pois_wf[key + ["pH", "pD", "pA", "pO25", "pBTTS"]], on=key, how="inner")
    if d.empty:
        return pd.DataFrame(), pd.DataFrame(), d
    res, over, btts = d["res"].to_numpy(), d["over"].to_numpy(), d["btts"].to_numpy()

    def methods(df):
        m = {"Poisson (statistical)": (df["pH"], df["pD"], df["pA"], df["pO25"], df["pBTTS"]),
             "ML": (df["mH"], df["mD"], df["mA"], df["mO"], df["mB"])}
        for w in weights:
            m[f"Blend (ML weight {w:.0%})"] = tuple(w * df[a] + (1 - w) * df[b] for a, b in
                                                    (("mH", "pH"), ("mD", "pD"), ("mA", "pA"), ("mO", "pO25"), ("mB", "pBTTS")))
        return m

    rows = {}
    for name, p in methods(d).items():
        rows[name] = _score(*[np.asarray(x) for x in p], res, over, btts)
    freq = np.bincount(res, minlength=3) / len(res)
    rows["Reference (constant frequencies)"] = _score(np.full(len(d), freq[0]), np.full(len(d), freq[1]), np.full(len(d), freq[2]),
                                                      np.full(len(d), over.mean()), np.full(len(d), btts.mean()), res, over, btts)
    tab_all = pd.DataFrame(rows).T; tab_all.insert(0, "Matches", len(d))

    sub = d.dropna(subset=["bH", "bD", "bA"])
    tab_bk = pd.DataFrame()
    if len(sub) >= 30:
        r2 = {}
        for name, p in methods(sub).items():
            r2[name] = _score(*[np.asarray(x) for x in p], sub["res"].to_numpy(), sub["over"].to_numpy(), sub["btts"].to_numpy())
        bo = sub["bO"].fillna(sub["over"].mean()).to_numpy()
        r2["Bookmaker (margin removed)"] = _score(sub["bH"].to_numpy(), sub["bD"].to_numpy(), sub["bA"].to_numpy(), bo,
                                                  np.full(len(sub), btts.mean()), sub["res"].to_numpy(), sub["over"].to_numpy(),
                                                  sub["btts"].to_numpy())
        tab_bk = pd.DataFrame(r2).T; tab_bk.insert(0, "Matches", len(sub))
        tab_bk.loc["Bookmaker (margin removed)", ["O2.5 log loss", "BTTS log loss"]] = np.nan  # not comparable (no BTTS odds / partial O2.5)
    return tab_all, tab_bk, d


def feature_importance(history: pd.DataFrame, prefs, use_odds: bool = False, holdout: float = 0.2, repeats: int = 3) -> pd.DataFrame:
    """Permutation importance (drop in 1X2 log loss when a feature is shuffled) on a held-out final slice."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    fb = FeatureBuilder(prefs=prefs, use_odds=use_odds)
    X, meta = fb.fit_transform(history)
    n = len(X); cut = int(n * (1 - holdout))
    clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=3, min_samples_leaf=40,
                                         l2_regularization=2.0, random_state=0).fit(X.iloc[:cut], meta["res"].iloc[:cut])
    Xt, yt = X.iloc[cut:], meta["res"].iloc[cut:]
    r = permutation_importance(clf, Xt, yt, scoring="neg_log_loss", n_repeats=repeats, random_state=0)
    out = pd.DataFrame({"Feature": X.columns, "Importance": r.importances_mean}).sort_values("Importance", ascending=False)
    return out.reset_index(drop=True)


def paired_test(d: pd.DataFrame, weights=(0.25, 0.5, 0.75)) -> pd.DataFrame:
    """Per-match 1X2 log-loss improvement over Poisson, with a standard error.
    Positive = better than Poisson.  |improvement| < 2 standard errors is indistinguishable from luck."""
    n = len(d); res = d["res"].to_numpy(); idx = np.arange(n)
    ll = lambda P: -np.log(np.clip(np.asarray(P)[idx, res], 1e-12, 1))
    norm = lambda P: P / P.sum(axis=1, keepdims=True)
    Pp = norm(d[["pH", "pD", "pA"]].to_numpy()); Pm = norm(d[["mH", "mD", "mA"]].to_numpy())
    base = ll(Pp)
    cands = {"ML": Pm, **{f"Blend (ML weight {w:.0%})": norm(w * Pm + (1 - w) * Pp) for w in weights}}
    rows = []
    for name, P in cands.items():
        diff = base - ll(P); se = float(diff.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
        m = float(diff.mean())
        verdict = ("Better – statistically meaningful" if m > 2 * se else "Better, but within noise" if m > 0
                   else "Worse than Poisson" if m < -2 * se else "No better than Poisson")
        rows.append({"Candidate": name, "Improvement (log loss)": m, "Std error": se, "Verdict": verdict})
    return pd.DataFrame(rows)
