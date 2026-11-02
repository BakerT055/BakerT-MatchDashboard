"""Pure helpers that turn predictions into coloured tables (no Streamlit imports here)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import TIERS, Settings, ODDS_LABELS
from .markets import MARKETS, MARKET_BY_KEY


def tier_index(p: float, s: Settings) -> int:
    if p >= s.tier_very_strong: return 0
    if p >= s.tier_strong: return 1
    if p >= s.tier_lean: return 2
    if p >= s.tier_weak: return 3
    return 4


def tier_name(p, s) -> str:
    return TIERS[tier_index(p, s)][0] if p == p else ""


def tier_css(p, s) -> str:
    if p != p:
        return ""
    _, bg, fg = TIERS[tier_index(p, s)]
    return f"background-color:{bg};color:{fg};font-weight:600"


def fair(p: float):
    return 1.0 / p if p and p > 0 else np.nan


def market_df(pred, keys: list[str], s: Settings) -> pd.DataFrame:
    rows = [{"Market": MARKET_BY_KEY[k].label, "Prob": pred.markets[k], "Fair odds": fair(pred.markets[k]),
             "Signal": tier_name(pred.markets[k], s)} for k in keys if k in pred.markets]
    return pd.DataFrame(rows)


def keys_for_groups(groups: list[str]) -> dict[str, list[str]]:
    return {g: [m.key for m in MARKETS if m.group == g] for g in groups}


def style_market(df: pd.DataFrame, s: Settings):
    def row(r):
        c = tier_css(r["Prob"], s)
        return ["", c, c, c]
    return df.style.apply(row, axis=1).format({"Prob": "{:.1%}", "Fair odds": "{:.2f}"}).hide(axis="index")


def ranked_df(pred, s: Settings, n: int = 10, bottom: bool = False) -> pd.DataFrame:
    items = [(MARKET_BY_KEY[k].label, p) for k, p in pred.markets.items() if MARKET_BY_KEY[k].rank]
    items.sort(key=lambda x: x[1], reverse=not bottom)
    rows = [{"Market": f"{i + 1}. {l}", "Prob": p, "Fair odds": fair(p), "Signal": tier_name(p, s)}
            for i, (l, p) in enumerate(items[:n])]
    return pd.DataFrame(rows)


def top_scores_df(pred) -> pd.DataFrame:
    return pd.DataFrame([{"Score": f"{h}-{a}", "Prob": p, "Fair odds": fair(p)} for h, a, p in pred.top_scores])


def _lerp(c1, c2, t):
    return "#" + "".join(f"{int(round(a + (b - a) * t)):02X}" for a, b in zip(c1, c2))


def heat_style(df: pd.DataFrame, fmt="{:.1%}"):
    lo, hi = (255, 255, 255), (62, 155, 82)
    mx = float(np.nanmax(df.to_numpy())) or 1.0
    return df.style.map(lambda v: f"background-color:{_lerp(lo, hi, min(1.0, v / mx))}" if v == v else "").format(fmt)


def scoreline_df(pred, size: int = 6) -> pd.DataFrame:
    g = pred.grid[:size, :size]
    df = pd.DataFrame(g, index=[f"Home {i}" for i in range(size)], columns=[str(j) for j in range(size)])
    df.index.name = "Home ↓ / Away →"
    return df


def value_df(pred, odds: dict, s: Settings) -> pd.DataFrame:
    rows = []
    for m in MARKETS:
        if not m.odds:
            continue
        o = odds.get(m.odds)
        p = pred.markets[m.key]
        has = o is not None and o == o and o > 1.0
        edge = p * o - 1 if has else np.nan
        verdict = "–" if not has else ("VALUE" if edge >= s.edge_min else ("Small +" if edge > 0 else "No value"))
        rows.append({"Market": m.label, "Model prob": p, "Fair odds": fair(p), "Book odds": o if has else np.nan,
                     "Book implied": 1 / o if has else np.nan, "Edge": edge, "Verdict": verdict})
    return pd.DataFrame(rows)


def margin_1x2(odds: dict):
    o = [odds.get(k) for k in ("oh", "od", "oa")]
    if all(v is not None and v == v and v > 1 for v in o):
        return sum(1 / v for v in o) - 1
    return None


_VERDICT = {"VALUE": ("#1E7B34", "#FFFFFF"), "Small +": ("#C6E7C1", "#1B5E20"), "No value": ("#F4C7C3", "#9C1C1C")}


def style_value(df: pd.DataFrame):
    def row(r):
        bg, fg = _VERDICT.get(r["Verdict"], ("", ""))
        c = f"background-color:{bg};color:{fg};font-weight:600" if bg else ""
        return ["", "", "", "", "", c, c]
    return (df.style.apply(row, axis=1)
            .format({"Model prob": "{:.1%}", "Fair odds": "{:.2f}", "Book odds": "{:.2f}", "Book implied": "{:.1%}", "Edge": "{:+.1%}"}, na_rep="–")
            .hide(axis="index"))


def style_check(df: pd.DataFrame, s: Settings):
    def fmt(v, idx):
        if v != v: return "–"
        if idx == "Games": return f"{int(v)}"
        if idx.endswith("%"): return f"{v:.0%}"
        return f"{v:.2f}"
    def col(c):
        return [tier_css(v, s) if (i in ("Win %", "Over 2.5 %", "BTTS %") and v == v) else "" for i, v in c.items()]
    sty = df.style.apply(col, axis=0)
    for i in df.index:
        sty = sty.format(lambda v, i=i: fmt(v, i), subset=pd.IndexSlice[[i], :])
    return sty


def fixture_label(r) -> str:
    d = r["Date"].strftime("%d-%b") if pd.notna(r["Date"]) else "TBC"
    return f"{d} | {r['League']} | {r['HomeTeam']} vs {r['AwayTeam']}"


def batch_predictions(fixtures: pd.DataFrame, predict_fn, s: Settings, odds_fn) -> pd.DataFrame:
    """One row per fixture with the headline markets, value flags and top score.
    predict_fn(home, away, league, date, odds) -> Prediction"""
    rows = []
    for idx, r in fixtures.iterrows():
        odds = odds_fn(r)
        p = predict_fn(r["HomeTeam"], r["AwayTeam"], r["League"], r["Date"], odds)
        m = p.markets
        vdf = value_df(p, odds, s)
        flags = [f"{x.Market} ({x.Edge:+.0%})" for x in vdf.itertuples() if x.Verdict == "VALUE"]
        best_k = max(("home", "draw", "away"), key=lambda k: m[k])
        rows.append({"idx": idx, "Date": r["Date"], "League": r["League"], "Home": r["HomeTeam"], "Away": r["AwayTeam"],
                     "xG H": p.lam_h, "xG A": p.lam_a, "P(1)": m["home"], "P(X)": m["draw"], "P(2)": m["away"],
                     "P(O2.5)": m["o2.5"], "P(U2.5)": m["u2.5"], "P(BTTS)": m["btts_y"],
                     "Top score": f"{p.top_scores[0][0]}-{p.top_scores[0][1]}", "Pick 1X2": MARKET_BY_KEY[best_k].label,
                     "Pick prob": m[best_k], "Engine": p.source, "VALUE flags": "; ".join(flags), "Low sample": "yes" if p.warnings else ""})
    return pd.DataFrame(rows)


def strongest_picks(fixtures: pd.DataFrame, predict_fn, s: Settings, min_prob: float, odds_fn=None, top_n: int = 50) -> pd.DataFrame:
    rows = []
    for _, r in fixtures.iterrows():
        p = predict_fn(r["HomeTeam"], r["AwayTeam"], r["League"], r["Date"], odds_fn(r) if odds_fn else None)
        for k, v in p.markets.items():
            if MARKET_BY_KEY[k].rank and v >= min_prob:
                rows.append({"Date": r["Date"], "League": r["League"], "Match": f"{r['HomeTeam']} vs {r['AwayTeam']}",
                             "Market": MARKET_BY_KEY[k].label, "Prob": v, "Fair odds": fair(v), "Signal": tier_name(v, s)})
    out = pd.DataFrame(rows, columns=["Date", "League", "Match", "Market", "Prob", "Fair odds", "Signal"])
    return out.sort_values("Prob", ascending=False).head(top_n).reset_index(drop=True)
